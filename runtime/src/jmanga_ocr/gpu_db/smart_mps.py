"""Metal kernels for conservative box expansion and reading order."""

from __future__ import annotations

from functools import lru_cache

import torch

_SMART_MPS_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

kernel void adaptive_expand(
    const device float* probability [[buffer(0)]],
    const device float* boxes [[buffer(1)]],
    const device uchar* valid [[buffer(2)]],
    device float* output_boxes [[buffer(3)]],
    device uchar* expanded_sides [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant uint& height [[buffer(6)]],
    constant uint& capacity [[buffer(7)]],
    constant uint& total_slots [[buffer(8)]],
    constant float& threshold [[buffer(9)]],
    constant float& maximum_support_area_ratio [[buffer(10)]],
    constant uint& minimum_pixels [[buffer(11)]],
    constant float& maximum_gap [[buffer(12)]],
    constant float& minimum_extent [[buffer(13)]],
    constant float& support_margin [[buffer(14)]],
    constant float& cross_cap_ratio [[buffer(15)]],
    constant float& end_cap_ratio [[buffer(16)]],
    constant float& minimum_cap [[buffer(17)]],
    constant float& maximum_cap [[buffer(18)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  if (group >= total_slots) return;
  uint box_base = group * 8;
  if (tid < 8) output_boxes[box_base + tid] = boxes[box_base + tid];
  if (tid < 4) expanded_sides[group * 4 + tid] = 0;
  if (!valid[group]) return;

  float2 corners[4];
  for (uint corner = 0; corner < 4; ++corner) {
    corners[corner] = float2(
        boxes[box_base + corner * 2], boxes[box_base + corner * 2 + 1]);
  }
  float2 centre = (corners[0] + corners[1] + corners[2] + corners[3]) * 0.25f;
  float2 horizontal = (corners[1] - corners[0]) + (corners[2] - corners[3]);
  float2 vertical = (corners[3] - corners[0]) + (corners[2] - corners[1]);
  horizontal /= max(length(horizontal), 1.0e-6f);
  vertical /= max(length(vertical), 1.0e-6f);

  float minimum_u = INFINITY, maximum_u = -INFINITY;
  float minimum_v = INFINITY, maximum_v = -INFINITY;
  for (uint corner = 0; corner < 4; ++corner) {
    float2 relative = corners[corner] - centre;
    float u = dot(relative, horizontal);
    float v = dot(relative, vertical);
    minimum_u = min(minimum_u, u);
    maximum_u = max(maximum_u, u);
    minimum_v = min(minimum_v, v);
    maximum_v = max(maximum_v, v);
  }
  float box_width = maximum_u - minimum_u;
  float box_height = maximum_v - minimum_v;
  float short_side = min(box_width, box_height);
  bool vertical_text = box_height > box_width * 1.05f;
  float cross_cap = clamp(cross_cap_ratio * short_side, minimum_cap, maximum_cap);
  float end_cap = clamp(end_cap_ratio * short_side, minimum_cap, maximum_cap);
  float caps[4] = {
      vertical_text ? cross_cap : end_cap,
      vertical_text ? cross_cap : end_cap,
      vertical_text ? end_cap : cross_cap,
      vertical_text ? end_cap : cross_cap};

  threadgroup float minimum_distances[1024];
  threadgroup float maximum_distances[1024];
  threadgroup uint support_counts[1024];
  for (uint side = 0; side < 4; ++side) {
    minimum_distances[side * 256 + tid] = INFINITY;
    maximum_distances[side * 256 + tid] = -INFINITY;
    support_counts[side * 256 + tid] = 0;
  }

  uint plane = width * height;
  uint batch = group / capacity;
  uint probability_offset = batch * plane;
  for (uint local = tid; local < plane; local += 256) {
    if (probability[probability_offset + local] < threshold) continue;
    float2 relative = float2(float(local % width), float(local / width)) - centre;
    float u = dot(relative, horizontal);
    float v = dot(relative, vertical);
    float distances[4] = {minimum_u - u, u - maximum_u, minimum_v - v, v - maximum_v};
    float orthogonal[4] = {v, v, u, u};
    float orthogonal_minimum[4] = {minimum_v, minimum_v, minimum_u, minimum_u};
    float orthogonal_maximum[4] = {maximum_v, maximum_v, maximum_u, maximum_u};
    for (uint side = 0; side < 4; ++side) {
      float span = orthogonal_maximum[side] - orthogonal_minimum[side];
      bool in_corridor =
          orthogonal[side] >= orthogonal_minimum[side] + 0.05f * span &&
          orthogonal[side] <= orthogonal_maximum[side] - 0.05f * span;
      if (in_corridor && distances[side] > 0.0f && distances[side] <= caps[side]) {
        uint index = side * 256 + tid;
        minimum_distances[index] = min(minimum_distances[index], distances[side]);
        maximum_distances[index] = max(maximum_distances[index], distances[side]);
        support_counts[index] += 1;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      for (uint side = 0; side < 4; ++side) {
        uint target = side * 256 + tid;
        uint source = target + stride;
        minimum_distances[target] = min(minimum_distances[target], minimum_distances[source]);
        maximum_distances[target] = max(maximum_distances[target], maximum_distances[source]);
        support_counts[target] += support_counts[source];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid != 0) return;

  float changes[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float maximum_support_pixels = maximum_support_area_ratio * short_side * short_side;
  for (uint side = 0; side < 4; ++side) {
    uint index = side * 256;
    uint count = support_counts[index];
    if (count < minimum_pixels || float(count) > maximum_support_pixels ||
        minimum_distances[index] > maximum_gap ||
        maximum_distances[index] < minimum_extent) continue;
    changes[side] = min(maximum_distances[index] + support_margin, caps[side]);
    expanded_sides[group * 4 + side] = 1;
  }
  if (changes[0] == 0.0f && changes[1] == 0.0f &&
      changes[2] == 0.0f && changes[3] == 0.0f) return;

  minimum_u -= changes[0];
  maximum_u += changes[1];
  minimum_v -= changes[2];
  maximum_v += changes[3];
  float local_u[4] = {minimum_u, maximum_u, maximum_u, minimum_u};
  float local_v[4] = {minimum_v, minimum_v, maximum_v, maximum_v};
  for (uint corner = 0; corner < 4; ++corner) {
    float2 point = centre + local_u[corner] * horizontal + local_v[corner] * vertical;
    output_boxes[box_base + corner * 2] = clamp(point.x, 0.0f, float(width));
    output_boxes[box_base + corner * 2 + 1] = clamp(point.y, 0.0f, float(height));
  }
}

kernel void reading_order(
    const device float* boxes [[buffer(0)]],
    const device uchar* valid [[buffer(1)]],
    device int* order [[buffer(2)]],
    device int* track_ids [[buffer(3)]],
    device uchar* vertical_output [[buffer(4)]],
    device float* skew_output [[buffer(5)]],
    device int* count_output [[buffer(6)]],
    constant uint& capacity [[buffer(7)]],
    constant uint& batch_size [[buffer(8)]],
    constant uint& robust_mode [[buffer(9)]],
    uint batch [[thread_position_in_grid]]) {
  if (batch >= batch_size) return;
  uint offset = batch * capacity;
  int active[64];
  float cross_minimum[64], cross_maximum[64], cross_centre[64];
  float read_minimum[64], read_maximum[64], read_centre[64];
  float widths[64], heights[64], areas[64], orientation_weights[64];
  float2 horizontal_axes[64], vertical_axes[64];
  int count = 0;
  float orientation_vote = 0.0f;
  for (uint slot = 0; slot < capacity; ++slot) {
    order[offset + slot] = -1;
    track_ids[offset + slot] = -1;
    if (!valid[offset + slot]) continue;
    active[count] = int(slot);
    uint base = (offset + slot) * 8;
    float2 p0 = float2(boxes[base], boxes[base + 1]);
    float2 p1 = float2(boxes[base + 2], boxes[base + 3]);
    float2 p2 = float2(boxes[base + 4], boxes[base + 5]);
    float2 p3 = float2(boxes[base + 6], boxes[base + 7]);
    float2 horizontal = (p1 - p0) + (p2 - p3);
    float2 vertical = (p3 - p0) + (p2 - p1);
    float width = max(length(p1 - p0), 1.0e-6f);
    float height = max(length(p3 - p0), 1.0e-6f);
    horizontal_axes[count] = horizontal / max(length(horizontal), 1.0e-6f);
    vertical_axes[count] = vertical / max(length(vertical), 1.0e-6f);
    widths[count] = width;
    heights[count] = height;
    areas[count] = max(width * height, 1.0f);
    orientation_weights[count] = areas[count];
    if (robust_mode == 0) {
      orientation_vote += areas[count] * log((height + 1.0f) / (width + 1.0f));
    }
    ++count;
  }
  count_output[batch] = count;
  if (count == 0) {
    vertical_output[batch] = 0;
    skew_output[batch] = 0.0f;
    return;
  }
  if (robust_mode != 0) {
    float sorted_areas[64];
    for (int index = 0; index < count; ++index) sorted_areas[index] = areas[index];
    for (int first = 0; first < count - 1; ++first) {
      for (int second = first + 1; second < count; ++second) {
        if (sorted_areas[second] < sorted_areas[first]) {
          float temporary = sorted_areas[first];
          sorted_areas[first] = sorted_areas[second];
          sorted_areas[second] = temporary;
        }
      }
    }
    float median_area = sorted_areas[(count - 1) / 2];
    int reliable_count = 0;
    for (int index = 0; index < count; ++index) {
      float aspect = max(widths[index], heights[index]) /
                     max(min(widths[index], heights[index]), 1.0e-6f);
      bool reliable = aspect >= 1.25f && areas[index] >= 0.15f * median_area;
      orientation_weights[index] = reliable ? areas[index] : 0.0f;
      if (!reliable) continue;
      orientation_vote += orientation_weights[index] *
          log((heights[index] + 1.0f) / (widths[index] + 1.0f));
      ++reliable_count;
    }
    if (reliable_count == 0) {
      for (int index = 0; index < count; ++index) {
        orientation_weights[index] = areas[index];
        orientation_vote += areas[index] *
            log((heights[index] + 1.0f) / (widths[index] + 1.0f));
      }
    }
  }
  bool is_vertical = orientation_vote >= 0.0f;
  vertical_output[batch] = is_vertical ? 1 : 0;
  float2 read_axis = float2(0.0f);
  float area_sum = 0.0f;
  for (int index = 0; index < count; ++index) {
    float2 axis = is_vertical ? vertical_axes[index] : horizontal_axes[index];
    float sign_coordinate = is_vertical ? axis.y : axis.x;
    if (sign_coordinate < 0.0f) axis *= -1.0f;
    float axis_weight = robust_mode != 0 ? orientation_weights[index] : areas[index];
    if (axis_weight > 0.0f) {
      read_axis += axis * axis_weight;
      area_sum += axis_weight;
    }
  }
  if (area_sum <= 1.0e-6f) {
    for (int index = 0; index < count; ++index) {
      float2 axis = is_vertical ? vertical_axes[index] : horizontal_axes[index];
      float sign_coordinate = is_vertical ? axis.y : axis.x;
      if (sign_coordinate < 0.0f) axis *= -1.0f;
      read_axis += axis * areas[index];
      area_sum += areas[index];
    }
  }
  read_axis /= max(area_sum, 1.0e-6f);
  read_axis /= max(length(read_axis), 1.0e-6f);
  float2 cross_axis = is_vertical
      ? float2(read_axis.y, -read_axis.x)
      : float2(-read_axis.y, read_axis.x);
  skew_output[batch] = is_vertical
      ? atan2(read_axis.x, read_axis.y)
      : atan2(read_axis.y, read_axis.x);

  for (int index = 0; index < count; ++index) {
    uint base = (offset + uint(active[index])) * 8;
    float minimum_cross = INFINITY, maximum_cross = -INFINITY;
    float minimum_read = INFINITY, maximum_read = -INFINITY;
    float centre_cross = 0.0f, centre_read = 0.0f;
    for (uint corner = 0; corner < 4; ++corner) {
      float2 point = float2(boxes[base + corner * 2], boxes[base + corner * 2 + 1]);
      float cross = dot(point, cross_axis);
      float read = dot(point, read_axis);
      minimum_cross = min(minimum_cross, cross);
      maximum_cross = max(maximum_cross, cross);
      minimum_read = min(minimum_read, read);
      maximum_read = max(maximum_read, read);
      centre_cross += cross * 0.25f;
      centre_read += read * 0.25f;
    }
    cross_minimum[index] = minimum_cross;
    cross_maximum[index] = maximum_cross;
    cross_centre[index] = centre_cross;
    read_minimum[index] = minimum_read;
    read_maximum[index] = maximum_read;
    read_centre[index] = centre_read;
  }

  int parent[64];
  for (int index = 0; index < count; ++index) parent[index] = index;
  if (robust_mode != 0) {
    for (int index = 0; index < count; ++index) {
      int best_root = index;
      float best_distance = INFINITY;
      for (int candidate_root = 0; candidate_root < index; ++candidate_root) {
        if (parent[candidate_root] != candidate_root) continue;
        bool complete_link = true;
        for (int member = 0; member < index; ++member) {
          if (parent[member] != candidate_root) continue;
          float first_width = cross_maximum[index] - cross_minimum[index];
          float second_width = cross_maximum[member] - cross_minimum[member];
          float cross_overlap = max(
              0.0f, min(cross_maximum[index], cross_maximum[member]) -
              max(cross_minimum[index], cross_minimum[member]));
          float first_length = read_maximum[index] - read_minimum[index];
          float second_length = read_maximum[member] - read_minimum[member];
          float read_overlap = max(
              0.0f, min(read_maximum[index], read_maximum[member]) -
              max(read_minimum[index], read_minimum[member]));
          bool compatible =
              (cross_overlap >= 0.25f * min(first_width, second_width) ||
               abs(cross_centre[index] - cross_centre[member]) <=
                   0.65f * min(first_width, second_width)) &&
              read_overlap <= 0.35f * min(first_length, second_length);
          if (!compatible) {
            complete_link = false;
            break;
          }
        }
        float distance = abs(cross_centre[index] - cross_centre[candidate_root]);
        if (complete_link && distance < best_distance) {
          best_distance = distance;
          best_root = candidate_root;
        }
      }
      parent[index] = best_root;
    }
  } else {
    for (int first = 0; first < count; ++first) {
      for (int second = first + 1; second < count; ++second) {
        float cross_width_first = cross_maximum[first] - cross_minimum[first];
        float cross_width_second = cross_maximum[second] - cross_minimum[second];
        float cross_overlap = max(
            0.0f, min(cross_maximum[first], cross_maximum[second]) -
            max(cross_minimum[first], cross_minimum[second]));
        float read_length_first = read_maximum[first] - read_minimum[first];
        float read_length_second = read_maximum[second] - read_minimum[second];
        float read_overlap = max(
            0.0f, min(read_maximum[first], read_maximum[second]) -
            max(read_minimum[first], read_minimum[second]));
        float read_gap = max(
            0.0f, max(read_minimum[first], read_minimum[second]) -
            min(read_maximum[first], read_maximum[second]));
        bool same_track =
            (cross_overlap >= 0.35f * min(cross_width_first, cross_width_second) ||
             abs(cross_centre[first] - cross_centre[second]) <=
                 0.50f * min(cross_width_first, cross_width_second)) &&
            read_overlap <= 0.20f * min(read_length_first, read_length_second) &&
            read_gap <= 0.75f * min(read_length_first, read_length_second);
        if (!same_track) continue;
        int root_first = first;
        while (parent[root_first] != root_first) root_first = parent[root_first];
        int root_second = second;
        while (parent[root_second] != root_second) root_second = parent[root_second];
        if (root_first != root_second) parent[root_second] = root_first;
      }
    }
  }
  for (int index = 0; index < count; ++index) {
    int root = index;
    while (parent[root] != root) root = parent[root];
    parent[index] = root;
  }

  int roots[64], track_count = 0;
  float track_cross[64];
  for (int index = 0; index < count; ++index) {
    int position = -1;
    for (int track = 0; track < track_count; ++track) {
      if (roots[track] == parent[index]) position = track;
    }
    if (position >= 0) continue;
    roots[track_count] = parent[index];
    float sum = 0.0f;
    int members = 0;
    for (int other = 0; other < count; ++other) {
      if (parent[other] == parent[index]) {
        sum += cross_centre[other];
        ++members;
      }
    }
    track_cross[track_count] = sum / float(max(members, 1));
    ++track_count;
  }
  for (int first = 0; first < track_count - 1; ++first) {
    for (int second = first + 1; second < track_count; ++second) {
      bool swap = is_vertical
          ? track_cross[second] > track_cross[first]
          : track_cross[second] < track_cross[first];
      if (swap) {
        float cross_temporary = track_cross[first];
        track_cross[first] = track_cross[second];
        track_cross[second] = cross_temporary;
        int root_temporary = roots[first];
        roots[first] = roots[second];
        roots[second] = root_temporary;
      }
    }
  }
  int output = 0;
  for (int track = 0; track < track_count; ++track) {
    int members[64], member_count = 0;
    for (int index = 0; index < count; ++index) {
      if (parent[index] == roots[track]) members[member_count++] = index;
    }
    for (int first = 0; first < member_count - 1; ++first) {
      for (int second = first + 1; second < member_count; ++second) {
        if (read_centre[members[second]] < read_centre[members[first]]) {
          int temporary = members[first];
          members[first] = members[second];
          members[second] = temporary;
        }
      }
    }
    for (int member = 0; member < member_count; ++member) {
      int active_index = members[member];
      int slot = active[active_index];
      order[offset + output] = slot;
      track_ids[offset + slot] = track;
      ++output;
    }
  }
}
"""


@lru_cache(maxsize=1)
def _library():
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is unavailable")
    return torch.mps.compile_shader(_SMART_MPS_SOURCE)


def adaptive_expand_mps(
    probability: torch.Tensor,
    boxes: torch.Tensor,
    valid: torch.Tensor,
    *,
    threshold: float,
    maximum_support_area_ratio: float,
    minimum_pixels: int,
    maximum_gap: float,
    minimum_extent: float,
    support_margin: float,
    cross_cap_ratio: float,
    end_cap_ratio: float,
    minimum_cap: float,
    maximum_cap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand valid quadrilaterals toward nearby weak DET support on MPS."""
    if probability.device.type != "mps":
        raise ValueError("adaptive_expand_mps requires MPS tensors")
    source = probability.contiguous().float()
    source_boxes = boxes.contiguous().float()
    source_valid = valid.contiguous()
    batch, _, height, width = source.shape
    capacity = source_boxes.shape[1]
    total_slots = batch * capacity
    output = torch.empty_like(source_boxes)
    expanded = torch.empty(
        batch, capacity, 4, device=source.device, dtype=torch.bool
    )
    _library().adaptive_expand(
        source,
        source_boxes,
        source_valid,
        output,
        expanded,
        width,
        height,
        capacity,
        total_slots,
        threshold,
        maximum_support_area_ratio,
        minimum_pixels,
        maximum_gap,
        minimum_extent,
        support_margin,
        cross_cap_ratio,
        end_cap_ratio,
        minimum_cap,
        maximum_cap,
        threads=[total_slots * 256, 1, 1],
        group_size=[256, 1, 1],
    )
    return output, expanded


def reading_order_mps(
    boxes: torch.Tensor,
    valid: torch.Tensor,
    *,
    robust: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Infer global orientation, skew, tracks and order on MPS."""
    if boxes.device.type != "mps":
        raise ValueError("reading_order_mps requires MPS tensors")
    source_boxes = boxes.contiguous().float()
    source_valid = valid.contiguous()
    batch, capacity = source_valid.shape
    order = torch.empty(batch, capacity, device=boxes.device, dtype=torch.int32)
    track_ids = torch.empty_like(order)
    vertical = torch.empty(batch, device=boxes.device, dtype=torch.bool)
    skew = torch.empty(batch, device=boxes.device, dtype=torch.float32)
    count = torch.empty(batch, device=boxes.device, dtype=torch.int32)
    _library().reading_order(
        source_boxes,
        source_valid,
        order,
        track_ids,
        vertical,
        skew,
        count,
        capacity,
        batch,
        int(robust),
        threads=[batch, 1, 1],
        group_size=[1, 1, 1],
    )
    return order, track_ids, vertical, skew, count
