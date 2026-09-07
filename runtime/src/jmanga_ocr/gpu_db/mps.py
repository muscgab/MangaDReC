"""Apple Metal connected-component labelling for DB probability maps."""

from __future__ import annotations

from functools import lru_cache

import torch

_MPS_CCL_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

kernel void init_labels(
    const device float* probability [[buffer(0)]],
    device int* parent [[buffer(1)]],
    constant float& threshold [[buffer(2)]],
    constant uint& total [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= total) return;
  parent[gid] = probability[gid] > threshold ? int(gid) : -1;
}

inline int root_of(device atomic_int* parent, int value, uint total) {
  int current = value;
  for (uint guard = 0;
       guard < 128 && current >= 0 && uint(current) < total;
       ++guard) {
    int next = atomic_load_explicit(&parent[current], memory_order_relaxed);
    if (next == current || next < 0) return current;
    current = next;
  }
  return current;
}

inline void union_pair(
    device atomic_int* parent, int first, int second, uint total) {
  int root_first = root_of(parent, first, total);
  int root_second = root_of(parent, second, total);
  for (uint guard = 0;
       guard < 32 && root_first != root_second &&
       root_first >= 0 && root_second >= 0;
       ++guard) {
    int high = max(root_first, root_second);
    int low = min(root_first, root_second);
    int previous = atomic_fetch_min_explicit(
        &parent[high], low, memory_order_relaxed);
    if (previous == high) return;
    root_first = root_of(parent, low, total);
    root_second = root_of(parent, previous, total);
  }
}

kernel void hook_labels(
    device atomic_int* parent [[buffer(0)]],
    constant uint& width [[buffer(1)]],
    constant uint& height [[buffer(2)]],
    constant uint& total [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= total) return;
  if (atomic_load_explicit(&parent[gid], memory_order_relaxed) < 0) return;

  uint plane = width * height;
  uint local = gid % plane;
  uint x = local % width;
  uint y = local / width;

  if (x + 1 < width &&
      atomic_load_explicit(&parent[gid + 1], memory_order_relaxed) >= 0) {
    union_pair(parent, int(gid), int(gid + 1), total);
  }
  if (y + 1 >= height) return;

  uint below = gid + width;
  if (atomic_load_explicit(&parent[below], memory_order_relaxed) >= 0) {
    union_pair(parent, int(gid), int(below), total);
  }
  if (x + 1 < width &&
      atomic_load_explicit(&parent[below + 1], memory_order_relaxed) >= 0) {
    union_pair(parent, int(gid), int(below + 1), total);
  }
  if (x > 0 &&
      atomic_load_explicit(&parent[below - 1], memory_order_relaxed) >= 0) {
    union_pair(parent, int(gid), int(below - 1), total);
  }
}

kernel void compress_labels(
    device int* parent [[buffer(0)]],
    constant uint& total [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= total || parent[gid] < 0) return;
  int current = parent[gid];
  for (uint guard = 0; guard < 512; ++guard) {
    int next = parent[current];
    if (next == current || next < 0) break;
    current = next;
  }
  parent[gid] = current;
}

kernel void count_roots(
    const device int* parent [[buffer(0)]],
    device atomic_int* counts [[buffer(1)]],
    constant uint& total [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= total) return;
  int root = parent[gid];
  if (root >= 0) {
    atomic_fetch_add_explicit(&counts[root], 1, memory_order_relaxed);
  }
}

kernel void select_top_components(
    const device int* counts [[buffer(0)]],
    device int* selected_roots [[buffer(1)]],
    device int* selected_sizes [[buffer(2)]],
    device int* raw_counts [[buffer(3)]],
    constant uint& plane [[buffer(4)]],
    constant uint& batch_size [[buffer(5)]],
    constant uint& capacity [[buffer(6)]],
    uint batch [[thread_position_in_grid]]) {
  if (batch >= batch_size) return;
  int best_sizes[64];
  int best_roots[64];
  for (uint slot = 0; slot < 64; ++slot) {
    best_sizes[slot] = 0;
    best_roots[slot] = -1;
  }

  uint offset = batch * plane;
  int raw_count = 0;
  for (uint local = 0; local < plane; ++local) {
    int size = counts[offset + local];
    if (size <= 0) continue;
    ++raw_count;
    int root = int(offset + local);
    uint position = capacity;
    for (uint slot = 0; slot < capacity; ++slot) {
      if (size > best_sizes[slot] ||
          (size == best_sizes[slot] && root < best_roots[slot])) {
        position = slot;
        break;
      }
    }
    if (position == capacity) continue;
    for (int slot = int(capacity) - 1; slot > int(position); --slot) {
      best_sizes[slot] = best_sizes[slot - 1];
      best_roots[slot] = best_roots[slot - 1];
    }
    best_sizes[position] = size;
    best_roots[position] = root;
  }
  raw_counts[batch] = raw_count;
  for (uint slot = 0; slot < capacity; ++slot) {
    selected_sizes[batch * capacity + slot] = best_sizes[slot];
    selected_roots[batch * capacity + slot] = best_roots[slot];
  }
}

kernel void component_moments(
    const device int* parent [[buffer(0)]],
    const device int* roots [[buffer(1)]],
    device float* angles [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant uint& height [[buffer(4)]],
    constant uint& capacity [[buffer(5)]],
    constant uint& total_slots [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  if (group >= total_slots) return;
  int root = roots[group];
  threadgroup float sum_x[256];
  threadgroup float sum_y[256];
  threadgroup float sum_xx[256];
  threadgroup float sum_yy[256];
  threadgroup float sum_xy[256];
  threadgroup float count[256];
  float local_x = 0.0f, local_y = 0.0f;
  float local_xx = 0.0f, local_yy = 0.0f, local_xy = 0.0f;
  float local_count = 0.0f;
  uint plane = width * height;
  uint batch = group / capacity;
  uint offset = batch * plane;
  if (root >= 0) {
    for (uint local = tid; local < plane; local += 256) {
      if (parent[offset + local] != root) continue;
      float x = float(local % width);
      float y = float(local / width);
      local_x += x;
      local_y += y;
      local_xx += x * x;
      local_yy += y * y;
      local_xy += x * y;
      local_count += 1.0f;
    }
  }
  sum_x[tid] = local_x;
  sum_y[tid] = local_y;
  sum_xx[tid] = local_xx;
  sum_yy[tid] = local_yy;
  sum_xy[tid] = local_xy;
  count[tid] = local_count;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sum_x[tid] += sum_x[tid + stride];
      sum_y[tid] += sum_y[tid + stride];
      sum_xx[tid] += sum_xx[tid + stride];
      sum_yy[tid] += sum_yy[tid + stride];
      sum_xy[tid] += sum_xy[tid + stride];
      count[tid] += count[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid != 0) return;
  if (count[0] <= 0.0f) {
    angles[group] = 0.0f;
    return;
  }
  float mean_x = sum_x[0] / count[0];
  float mean_y = sum_y[0] / count[0];
  float covariance_xx = sum_xx[0] / count[0] - mean_x * mean_x;
  float covariance_yy = sum_yy[0] / count[0] - mean_y * mean_y;
  float covariance_xy = sum_xy[0] / count[0] - mean_x * mean_y;
  angles[group] = 0.5f * atan2(
      2.0f * covariance_xy, covariance_xx - covariance_yy);
}

kernel void component_extents(
    const device int* parent [[buffer(0)]],
    const device int* roots [[buffer(1)]],
    const device float* principal_angles [[buffer(2)]],
    const device float* angle_offsets [[buffer(3)]],
    device float* selected_angles [[buffer(4)]],
    device float* extents [[buffer(5)]],
    constant uint& width [[buffer(6)]],
    constant uint& height [[buffer(7)]],
    constant uint& capacity [[buffer(8)]],
    constant uint& total_slots [[buffer(9)]],
    constant uint& angle_count [[buffer(10)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  if (group >= total_slots) return;
  int root = roots[group];
  threadgroup float minimum_u[256];
  threadgroup float maximum_u[256];
  threadgroup float minimum_v[256];
  threadgroup float maximum_v[256];
  threadgroup float best_area;
  threadgroup float best_angle;
  threadgroup float best_minimum_u;
  threadgroup float best_maximum_u;
  threadgroup float best_minimum_v;
  threadgroup float best_maximum_v;
  if (tid == 0) {
    best_area = INFINITY;
    best_angle = principal_angles[group];
    best_minimum_u = 0.0f;
    best_maximum_u = 0.0f;
    best_minimum_v = 0.0f;
    best_maximum_v = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint plane = width * height;
  uint batch = group / capacity;
  uint offset = batch * plane;
  for (uint candidate = 0; candidate < angle_count; ++candidate) {
    float angle = principal_angles[group] + angle_offsets[candidate];
    float cosine = cos(angle);
    float sine = sin(angle);
    float min_u = INFINITY, max_u = -INFINITY;
    float min_v = INFINITY, max_v = -INFINITY;
    if (root >= 0) {
      for (uint local = tid; local < plane; local += 256) {
        if (parent[offset + local] != root) continue;
        float x = float(local % width);
        float y = float(local / width);
        float u = cosine * x + sine * y;
        float v = -sine * x + cosine * y;
        min_u = min(min_u, u);
        max_u = max(max_u, u);
        min_v = min(min_v, v);
        max_v = max(max_v, v);
      }
    }
    minimum_u[tid] = min_u;
    maximum_u[tid] = max_u;
    minimum_v[tid] = min_v;
    maximum_v[tid] = max_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
      if (tid < stride) {
        minimum_u[tid] = min(minimum_u[tid], minimum_u[tid + stride]);
        maximum_u[tid] = max(maximum_u[tid], maximum_u[tid + stride]);
        minimum_v[tid] = min(minimum_v[tid], minimum_v[tid + stride]);
        maximum_v[tid] = max(maximum_v[tid], maximum_v[tid + stride]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
      float area = (maximum_u[0] - minimum_u[0]) *
                   (maximum_v[0] - minimum_v[0]);
      if (area < best_area) {
        best_area = area;
        best_angle = angle;
        best_minimum_u = minimum_u[0];
        best_maximum_u = maximum_u[0];
        best_minimum_v = minimum_v[0];
        best_maximum_v = maximum_v[0];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid != 0) return;
  selected_angles[group] = best_angle;
  extents[group * 4] = best_minimum_u;
  extents[group * 4 + 1] = best_maximum_u;
  extents[group * 4 + 2] = best_minimum_v;
  extents[group * 4 + 3] = best_maximum_v;
}

kernel void score_and_box(
    const device float* probability [[buffer(0)]],
    const device int* roots [[buffer(1)]],
    const device int* sizes [[buffer(2)]],
    const device float* angles [[buffer(3)]],
    const device float* extents [[buffer(4)]],
    device float* boxes [[buffer(5)]],
    device float* scores [[buffer(6)]],
    device uchar* valid [[buffer(7)]],
    device atomic_int* output_counts [[buffer(8)]],
    constant uint& width [[buffer(9)]],
    constant uint& height [[buffer(10)]],
    constant uint& capacity [[buffer(11)]],
    constant uint& total_slots [[buffer(12)]],
    constant float& box_threshold [[buffer(13)]],
    constant float& unclip_ratio [[buffer(14)]],
    constant float& min_size [[buffer(15)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  if (group >= total_slots) return;
  threadgroup float score_sums[256];
  threadgroup int score_counts[256];
  float score_sum = 0.0f;
  int score_count = 0;
  int root = roots[group];
  float minimum_u = extents[group * 4];
  float maximum_u = extents[group * 4 + 1];
  float minimum_v = extents[group * 4 + 2];
  float maximum_v = extents[group * 4 + 3];
  float angle = angles[group];
  float cosine = cos(angle);
  float sine = sin(angle);
  uint plane = width * height;
  uint batch = group / capacity;
  uint offset = batch * plane;
  if (root >= 0 && sizes[group] > 0) {
    for (uint local = tid; local < plane; local += 256) {
      float x = float(local % width);
      float y = float(local / width);
      float u = cosine * x + sine * y;
      float v = -sine * x + cosine * y;
      if (u >= minimum_u && u <= maximum_u &&
          v >= minimum_v && v <= maximum_v) {
        score_sum += probability[offset + local];
        ++score_count;
      }
    }
  }
  score_sums[tid] = score_sum;
  score_counts[tid] = score_count;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      score_sums[tid] += score_sums[tid + stride];
      score_counts[tid] += score_counts[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid != 0) return;

  uint box_base = group * 8;
  for (uint index = 0; index < 8; ++index) boxes[box_base + index] = 0.0f;
  scores[group] = 0.0f;
  valid[group] = 0;
  if (root < 0 || sizes[group] <= 0) return;
  float box_width = maximum_u - minimum_u;
  float box_height = maximum_v - minimum_v;
  if (min(box_width, box_height) < min_size) return;
  float score = score_sums[0] / float(max(score_counts[0], 1));
  if (score < box_threshold) return;

  float distance = box_width * box_height * unclip_ratio /
                   max(2.0f * (box_width + box_height), 1.0e-6f);
  minimum_u -= distance;
  maximum_u += distance;
  minimum_v -= distance;
  maximum_v += distance;
  if (min(maximum_u - minimum_u, maximum_v - minimum_v) <
      min_size + 2.0f) return;

  float corners_u[4] = {minimum_u, maximum_u, maximum_u, minimum_u};
  float corners_v[4] = {minimum_v, minimum_v, maximum_v, maximum_v};
  float corner_x[4];
  float corner_y[4];
  for (uint corner = 0; corner < 4; ++corner) {
    corner_x[corner] = clamp(
        corners_u[corner] * cosine - corners_v[corner] * sine,
        0.0f, float(width));
    corner_y[corner] = clamp(
        corners_u[corner] * sine + corners_v[corner] * cosine,
        0.0f, float(height));
  }
  int order[4] = {0, 1, 2, 3};
  for (uint first = 0; first < 3; ++first) {
    for (uint second = first + 1; second < 4; ++second) {
      int left = order[first];
      int right = order[second];
      if (corner_x[right] < corner_x[left] ||
          (corner_x[right] == corner_x[left] &&
           corner_y[right] < corner_y[left])) {
        order[first] = right;
        order[second] = left;
      }
    }
  }
  if (corner_y[order[0]] > corner_y[order[1]]) {
    int temporary = order[0];
    order[0] = order[1];
    order[1] = temporary;
  }
  if (corner_y[order[2]] > corner_y[order[3]]) {
    int temporary = order[2];
    order[2] = order[3];
    order[3] = temporary;
  }
  int canonical[4] = {order[0], order[2], order[3], order[1]};
  for (uint corner = 0; corner < 4; ++corner) {
    boxes[box_base + corner * 2] = corner_x[canonical[corner]];
    boxes[box_base + corner * 2 + 1] = corner_y[canonical[corner]];
  }
  scores[group] = score;
  valid[group] = 1;
  atomic_fetch_add_explicit(
      &output_counts[batch], 1, memory_order_relaxed);
}
"""


@lru_cache(maxsize=1)
def _library():
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is unavailable")
    return torch.mps.compile_shader(_MPS_CCL_SOURCE)


def connected_components_mps(
    probability: torch.Tensor,
    threshold: float,
    rounds: int,
) -> torch.Tensor:
    """Return global root labels for an MPS ``[B,1,H,W]`` probability map."""
    if probability.device.type != "mps":
        raise ValueError("connected_components_mps requires an MPS tensor")
    if probability.ndim != 4 or probability.shape[1] != 1:
        raise ValueError("probability must have shape [B,1,H,W]")
    if rounds < 1:
        raise ValueError("rounds must be positive")

    source = probability.contiguous().float()
    batch, _, height, width = source.shape
    total = batch * height * width
    parent = torch.empty(total, device=source.device, dtype=torch.int32)
    library = _library()
    dispatch = {
        "threads": [total, 1, 1],
        "group_size": [min(256, total), 1, 1],
    }
    library.init_labels(source, parent, threshold, total, **dispatch)
    for _ in range(rounds):
        library.hook_labels(parent, width, height, total, **dispatch)
        library.compress_labels(parent, total, **dispatch)
    return parent.view(batch, height, width)


def decode_db_mps(
    probability: torch.Tensor,
    *,
    threshold: float,
    box_threshold: float,
    unclip_ratio: float,
    min_size: float,
    max_components: int,
    rounds: int,
    angle_offsets_radians: tuple[float, ...],
    angle_refine_offsets_radians: tuple[float, ...],
) -> tuple[torch.Tensor, ...]:
    """Run complete fixed-slot DB decoding without leaving Apple GPU memory."""
    if max_components > 64:
        raise ValueError("MPS DB decoder supports at most 64 components")
    source = probability.contiguous().float()
    batch, _, height, width = source.shape
    plane = height * width
    total = batch * plane
    library = _library()
    dispatch = {
        "threads": [total, 1, 1],
        "group_size": [min(256, total), 1, 1],
    }
    parent = torch.empty(total, device=source.device, dtype=torch.int32)
    library.init_labels(source, parent, threshold, total, **dispatch)
    for _ in range(rounds):
        library.hook_labels(parent, width, height, total, **dispatch)
        library.compress_labels(parent, total, **dispatch)

    component_counts = torch.zeros_like(parent)
    library.count_roots(parent, component_counts, total, **dispatch)
    total_slots = batch * max_components
    roots = torch.empty(total_slots, device=source.device, dtype=torch.int32)
    sizes = torch.empty_like(roots)
    raw_counts = torch.empty(batch, device=source.device, dtype=torch.int32)
    library.select_top_components(
        component_counts,
        roots,
        sizes,
        raw_counts,
        plane,
        batch,
        max_components,
        threads=[batch, 1, 1],
        group_size=[1, 1, 1],
    )

    group_dispatch = {
        "threads": [total_slots * 256, 1, 1],
        "group_size": [256, 1, 1],
    }
    principal_angles = torch.empty(
        total_slots, device=source.device, dtype=torch.float32
    )
    library.component_moments(
        parent,
        roots,
        principal_angles,
        width,
        height,
        max_components,
        total_slots,
        **group_dispatch,
    )
    angle_offsets = torch.tensor(
        angle_offsets_radians, device=source.device, dtype=torch.float32
    )
    selected_angles = torch.empty_like(principal_angles)
    extents = torch.empty(
        total_slots, 4, device=source.device, dtype=torch.float32
    )
    library.component_extents(
        parent,
        roots,
        principal_angles,
        angle_offsets,
        selected_angles,
        extents,
        width,
        height,
        max_components,
        total_slots,
        len(angle_offsets_radians),
        **group_dispatch,
    )
    refine_angle_offsets = torch.tensor(
        angle_refine_offsets_radians, device=source.device, dtype=torch.float32
    )
    refined_angles = torch.empty_like(selected_angles)
    library.component_extents(
        parent,
        roots,
        selected_angles,
        refine_angle_offsets,
        refined_angles,
        extents,
        width,
        height,
        max_components,
        total_slots,
        len(angle_refine_offsets_radians),
        **group_dispatch,
    )
    boxes = torch.empty(
        total_slots, 4, 2, device=source.device, dtype=torch.float32
    )
    scores = torch.empty(total_slots, device=source.device, dtype=torch.float32)
    valid = torch.empty(total_slots, device=source.device, dtype=torch.bool)
    output_counts = torch.zeros(batch, device=source.device, dtype=torch.int32)
    library.score_and_box(
        source,
        roots,
        sizes,
        refined_angles,
        extents,
        boxes,
        scores,
        valid,
        output_counts,
        width,
        height,
        max_components,
        total_slots,
        box_threshold,
        unclip_ratio,
        min_size,
        **group_dispatch,
    )
    return (
        parent.view(batch, height, width),
        boxes.view(batch, max_components, 4, 2),
        scores.view(batch, max_components),
        valid.view(batch, max_components),
        output_counts,
        raw_counts,
    )
