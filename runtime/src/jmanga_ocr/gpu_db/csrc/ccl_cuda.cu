#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <climits>
#include <cmath>
#include <vector>

namespace {

__global__ void init_labels_kernel(
    const float* probability, int* parent, float threshold, int total) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= total) return;
  parent[gid] = probability[gid] > threshold ? gid : -1;
}

__device__ int root_of(int* parent, int value, int total) {
  int current = value;
  for (int guard = 0;
       guard < 128 && current >= 0 && current < total;
       ++guard) {
    int next = parent[current];
    if (next == current || next < 0) return current;
    current = next;
  }
  return current;
}

__device__ void union_pair(int* parent, int first, int second, int total) {
  int root_first = root_of(parent, first, total);
  int root_second = root_of(parent, second, total);
  for (int guard = 0;
       guard < 32 && root_first != root_second &&
       root_first >= 0 && root_second >= 0;
       ++guard) {
    int high = max(root_first, root_second);
    int low = min(root_first, root_second);
    int previous = atomicMin(parent + high, low);
    if (previous == high) return;
    root_first = root_of(parent, low, total);
    root_second = root_of(parent, previous, total);
  }
}

__global__ void hook_labels_kernel(
    int* parent, int width, int height, int total) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= total || parent[gid] < 0) return;

  int plane = width * height;
  int local = gid % plane;
  int x = local % width;
  int y = local / width;

  if (x + 1 < width && parent[gid + 1] >= 0) {
    union_pair(parent, gid, gid + 1, total);
  }
  if (y + 1 >= height) return;

  int below = gid + width;
  if (parent[below] >= 0) union_pair(parent, gid, below, total);
  if (x + 1 < width && parent[below + 1] >= 0) {
    union_pair(parent, gid, below + 1, total);
  }
  if (x > 0 && parent[below - 1] >= 0) {
    union_pair(parent, gid, below - 1, total);
  }
}

__global__ void compress_labels_kernel(int* parent, int total) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= total || parent[gid] < 0) return;
  int current = parent[gid];
  for (int guard = 0; guard < 512; ++guard) {
    int next = parent[current];
    if (next == current || next < 0) break;
    current = next;
  }
  parent[gid] = current;
}

__global__ void count_roots_kernel(
    const int* parent, int* counts, int total) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= total) return;
  int root = parent[gid];
  if (root >= 0) atomicAdd(counts + root, 1);
}

__global__ void select_top_components_kernel(
    const int* counts,
    int* selected_roots,
    int* selected_sizes,
    int* raw_counts,
    int plane,
    int batch_size,
    int capacity) {
  int batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch >= batch_size) return;
  int best_sizes[64];
  int best_roots[64];
  for (int slot = 0; slot < 64; ++slot) {
    best_sizes[slot] = 0;
    best_roots[slot] = -1;
  }
  int offset = batch * plane;
  int raw_count = 0;
  for (int local = 0; local < plane; ++local) {
    int size = counts[offset + local];
    if (size <= 0) continue;
    ++raw_count;
    int root = offset + local;
    int position = capacity;
    for (int slot = 0; slot < capacity; ++slot) {
      if (size > best_sizes[slot] ||
          (size == best_sizes[slot] && root < best_roots[slot])) {
        position = slot;
        break;
      }
    }
    if (position == capacity) continue;
    for (int slot = capacity - 1; slot > position; --slot) {
      best_sizes[slot] = best_sizes[slot - 1];
      best_roots[slot] = best_roots[slot - 1];
    }
    best_sizes[position] = size;
    best_roots[position] = root;
  }
  raw_counts[batch] = raw_count;
  for (int slot = 0; slot < capacity; ++slot) {
    selected_sizes[batch * capacity + slot] = best_sizes[slot];
    selected_roots[batch * capacity + slot] = best_roots[slot];
  }
}

__global__ void component_moments_kernel(
    const int* parent,
    const int* roots,
    float* angles,
    int width,
    int height,
    int capacity,
    int total_slots) {
  int group = blockIdx.x;
  int tid = threadIdx.x;
  if (group >= total_slots) return;
  int root = roots[group];
  __shared__ float sum_x[256];
  __shared__ float sum_y[256];
  __shared__ float sum_xx[256];
  __shared__ float sum_yy[256];
  __shared__ float sum_xy[256];
  __shared__ float count[256];
  float local_x = 0.0f;
  float local_y = 0.0f;
  float local_xx = 0.0f;
  float local_yy = 0.0f;
  float local_xy = 0.0f;
  float local_count = 0.0f;
  int plane = width * height;
  int batch = group / capacity;
  int offset = batch * plane;
  if (root >= 0) {
    for (int local = tid; local < plane; local += blockDim.x) {
      if (parent[offset + local] != root) continue;
      float x = static_cast<float>(local % width);
      float y = static_cast<float>(local / width);
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
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sum_x[tid] += sum_x[tid + stride];
      sum_y[tid] += sum_y[tid + stride];
      sum_xx[tid] += sum_xx[tid + stride];
      sum_yy[tid] += sum_yy[tid + stride];
      sum_xy[tid] += sum_xy[tid + stride];
      count[tid] += count[tid + stride];
    }
    __syncthreads();
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
  angles[group] = 0.5f * atan2f(
      2.0f * covariance_xy, covariance_xx - covariance_yy);
}

__global__ void component_extents_kernel(
    const int* parent,
    const int* roots,
    const float* principal_angles,
    const float* angle_offsets,
    float* selected_angles,
    float* extents,
    int width,
    int height,
    int capacity,
    int total_slots,
    int angle_count) {
  int group = blockIdx.x;
  int tid = threadIdx.x;
  if (group >= total_slots) return;
  int root = roots[group];
  __shared__ float minimum_u[256];
  __shared__ float maximum_u[256];
  __shared__ float minimum_v[256];
  __shared__ float maximum_v[256];
  __shared__ float best_area;
  __shared__ float best_angle;
  __shared__ float best_minimum_u;
  __shared__ float best_maximum_u;
  __shared__ float best_minimum_v;
  __shared__ float best_maximum_v;
  if (tid == 0) {
    best_area = INFINITY;
    best_angle = principal_angles[group];
    best_minimum_u = 0.0f;
    best_maximum_u = 0.0f;
    best_minimum_v = 0.0f;
    best_maximum_v = 0.0f;
  }
  __syncthreads();

  int plane = width * height;
  int batch = group / capacity;
  int offset = batch * plane;
  for (int candidate = 0; candidate < angle_count; ++candidate) {
    float angle = principal_angles[group] + angle_offsets[candidate];
    float cosine = cosf(angle);
    float sine = sinf(angle);
    float min_u = INFINITY;
    float max_u = -INFINITY;
    float min_v = INFINITY;
    float max_v = -INFINITY;
    if (root >= 0) {
      for (int local = tid; local < plane; local += blockDim.x) {
        if (parent[offset + local] != root) continue;
        float x = static_cast<float>(local % width);
        float y = static_cast<float>(local / width);
        float u = cosine * x + sine * y;
        float v = -sine * x + cosine * y;
        min_u = fminf(min_u, u);
        max_u = fmaxf(max_u, u);
        min_v = fminf(min_v, v);
        max_v = fmaxf(max_v, v);
      }
    }
    minimum_u[tid] = min_u;
    maximum_u[tid] = max_u;
    minimum_v[tid] = min_v;
    maximum_v[tid] = max_v;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
      if (tid < stride) {
        minimum_u[tid] = fminf(minimum_u[tid], minimum_u[tid + stride]);
        maximum_u[tid] = fmaxf(maximum_u[tid], maximum_u[tid + stride]);
        minimum_v[tid] = fminf(minimum_v[tid], minimum_v[tid + stride]);
        maximum_v[tid] = fmaxf(maximum_v[tid], maximum_v[tid + stride]);
      }
      __syncthreads();
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
    __syncthreads();
  }
  if (tid != 0) return;
  selected_angles[group] = best_angle;
  extents[group * 4] = best_minimum_u;
  extents[group * 4 + 1] = best_maximum_u;
  extents[group * 4 + 2] = best_minimum_v;
  extents[group * 4 + 3] = best_maximum_v;
}

__global__ void score_and_box_kernel(
    const float* probability,
    const int* roots,
    const int* sizes,
    const float* angles,
    const float* extents,
    float* boxes,
    float* scores,
    bool* valid,
    int* output_counts,
    int width,
    int height,
    int capacity,
    int total_slots,
    float box_threshold,
    float unclip_ratio,
    float min_size) {
  int group = blockIdx.x;
  int tid = threadIdx.x;
  if (group >= total_slots) return;
  __shared__ float score_sums[256];
  __shared__ int score_counts[256];
  float score_sum = 0.0f;
  int score_count = 0;
  int root = roots[group];
  float minimum_u_value = extents[group * 4];
  float maximum_u_value = extents[group * 4 + 1];
  float minimum_v_value = extents[group * 4 + 2];
  float maximum_v_value = extents[group * 4 + 3];
  float angle = angles[group];
  float cosine = cosf(angle);
  float sine = sinf(angle);
  int plane = width * height;
  int batch = group / capacity;
  int offset = batch * plane;
  if (root >= 0 && sizes[group] > 0) {
    for (int local = tid; local < plane; local += blockDim.x) {
      float x = static_cast<float>(local % width);
      float y = static_cast<float>(local / width);
      float u = cosine * x + sine * y;
      float v = -sine * x + cosine * y;
      if (u >= minimum_u_value && u <= maximum_u_value &&
          v >= minimum_v_value && v <= maximum_v_value) {
        score_sum += probability[offset + local];
        ++score_count;
      }
    }
  }
  score_sums[tid] = score_sum;
  score_counts[tid] = score_count;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      score_sums[tid] += score_sums[tid + stride];
      score_counts[tid] += score_counts[tid + stride];
    }
    __syncthreads();
  }
  if (tid != 0) return;

  int box_base = group * 8;
  for (int index = 0; index < 8; ++index) boxes[box_base + index] = 0.0f;
  scores[group] = 0.0f;
  valid[group] = false;
  if (root < 0 || sizes[group] <= 0) return;
  float box_width = maximum_u_value - minimum_u_value;
  float box_height = maximum_v_value - minimum_v_value;
  if (fminf(box_width, box_height) < min_size) return;
  float score = score_sums[0] / static_cast<float>(max(score_counts[0], 1));
  if (score < box_threshold) return;

  float distance = box_width * box_height * unclip_ratio /
                   fmaxf(2.0f * (box_width + box_height), 1.0e-6f);
  minimum_u_value -= distance;
  maximum_u_value += distance;
  minimum_v_value -= distance;
  maximum_v_value += distance;
  if (fminf(
          maximum_u_value - minimum_u_value,
          maximum_v_value - minimum_v_value) < min_size + 2.0f) {
    return;
  }

  float corners_u[4] = {
      minimum_u_value, maximum_u_value, maximum_u_value, minimum_u_value};
  float corners_v[4] = {
      minimum_v_value, minimum_v_value, maximum_v_value, maximum_v_value};
  float corner_x[4];
  float corner_y[4];
  for (int corner = 0; corner < 4; ++corner) {
    float x = corners_u[corner] * cosine - corners_v[corner] * sine;
    float y = corners_u[corner] * sine + corners_v[corner] * cosine;
    corner_x[corner] = fminf(fmaxf(x, 0.0f), static_cast<float>(width));
    corner_y[corner] = fminf(fmaxf(y, 0.0f), static_cast<float>(height));
  }
  int order[4] = {0, 1, 2, 3};
  for (int first = 0; first < 3; ++first) {
    for (int second = first + 1; second < 4; ++second) {
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
  for (int corner = 0; corner < 4; ++corner) {
    boxes[box_base + corner * 2] = corner_x[canonical[corner]];
    boxes[box_base + corner * 2 + 1] = corner_y[canonical[corner]];
  }
  scores[group] = score;
  valid[group] = true;
  atomicAdd(output_counts + batch, 1);
}

torch::Tensor connected_components_impl(
    torch::Tensor probability, double threshold, int64_t rounds) {
  int total = static_cast<int>(probability.numel());
  int width = static_cast<int>(probability.size(3));
  int height = static_cast<int>(probability.size(2));
  auto parent = torch::empty(
      {probability.size(0), height, width},
      probability.options().dtype(torch::kInt32));
  constexpr int threads = 256;
  int blocks = (total + threads - 1) / threads;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  init_labels_kernel<<<blocks, threads, 0, stream>>>(
      probability.data_ptr<float>(), parent.data_ptr<int>(),
      static_cast<float>(threshold), total);
  for (int64_t round = 0; round < rounds; ++round) {
    hook_labels_kernel<<<blocks, threads, 0, stream>>>(
        parent.data_ptr<int>(), width, height, total);
    compress_labels_kernel<<<blocks, threads, 0, stream>>>(
        parent.data_ptr<int>(), total);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return parent;
}

torch::Tensor connected_components(
    torch::Tensor probability, double threshold, int64_t rounds) {
  TORCH_CHECK(probability.is_cuda(), "probability must be CUDA");
  TORCH_CHECK(probability.scalar_type() == torch::kFloat32,
              "probability must be float32");
  TORCH_CHECK(probability.dim() == 4 && probability.size(1) == 1,
              "probability must have shape [B,1,H,W]");
  TORCH_CHECK(probability.is_contiguous(), "probability must be contiguous");
  TORCH_CHECK(rounds > 0, "rounds must be positive");

  c10::cuda::CUDAGuard guard(probability.device());
  int64_t total64 = probability.numel();
  TORCH_CHECK(total64 <= INT_MAX, "probability map is too large");
  return connected_components_impl(probability, threshold, rounds);
}

std::vector<torch::Tensor> decode_db(
    torch::Tensor probability,
    double threshold,
    double box_threshold,
    double unclip_ratio,
    double min_size,
    int64_t max_components,
    int64_t rounds,
    torch::Tensor angle_offsets,
    torch::Tensor angle_refine_offsets) {
  TORCH_CHECK(probability.is_cuda(), "probability must be CUDA");
  TORCH_CHECK(probability.scalar_type() == torch::kFloat32,
              "probability must be float32");
  TORCH_CHECK(probability.dim() == 4 && probability.size(1) == 1,
              "probability must have shape [B,1,H,W]");
  TORCH_CHECK(probability.is_contiguous(), "probability must be contiguous");
  TORCH_CHECK(rounds > 0, "rounds must be positive");
  TORCH_CHECK(max_components > 0 && max_components <= 64,
              "max_components must be in [1,64]");
  TORCH_CHECK(angle_offsets.is_cuda() && angle_offsets.is_contiguous(),
              "angle_offsets must be a contiguous CUDA tensor");
  TORCH_CHECK(angle_offsets.scalar_type() == torch::kFloat32 &&
                  angle_offsets.dim() == 1 && angle_offsets.numel() > 0,
              "angle_offsets must be a non-empty float32 vector");
  TORCH_CHECK(angle_offsets.device() == probability.device(),
              "angle_offsets and probability must share a device");
  TORCH_CHECK(
      angle_refine_offsets.is_cuda() && angle_refine_offsets.is_contiguous(),
      "angle_refine_offsets must be a contiguous CUDA tensor");
  TORCH_CHECK(
      angle_refine_offsets.scalar_type() == torch::kFloat32 &&
          angle_refine_offsets.dim() == 1 &&
          angle_refine_offsets.numel() > 0,
      "angle_refine_offsets must be a non-empty float32 vector");
  TORCH_CHECK(angle_refine_offsets.device() == probability.device(),
              "angle_refine_offsets and probability must share a device");
  TORCH_CHECK(probability.numel() <= INT_MAX, "probability map is too large");

  c10::cuda::CUDAGuard guard(probability.device());
  int batch = static_cast<int>(probability.size(0));
  int height = static_cast<int>(probability.size(2));
  int width = static_cast<int>(probability.size(3));
  int plane = width * height;
  int total = batch * plane;
  int capacity = static_cast<int>(max_components);
  int total_slots = batch * capacity;
  int angle_count = static_cast<int>(angle_offsets.numel());
  int refine_angle_count = static_cast<int>(angle_refine_offsets.numel());
  auto int_options = probability.options().dtype(torch::kInt32);
  auto float_options = probability.options().dtype(torch::kFloat32);
  auto bool_options = probability.options().dtype(torch::kBool);
  auto parent = connected_components_impl(probability, threshold, rounds);
  auto component_counts = torch::zeros({total}, int_options);
  auto roots = torch::empty({batch, capacity}, int_options);
  auto sizes = torch::empty({batch, capacity}, int_options);
  auto raw_counts = torch::empty({batch}, int_options);
  auto angles = torch::empty({total_slots}, float_options);
  auto extents = torch::empty({total_slots, 4}, float_options);
  auto selected_angles = torch::empty({total_slots}, float_options);
  auto refined_angles = torch::empty({total_slots}, float_options);
  auto boxes = torch::empty({batch, capacity, 4, 2}, float_options);
  auto scores = torch::empty({batch, capacity}, float_options);
  auto valid = torch::empty({batch, capacity}, bool_options);
  auto output_counts = torch::zeros({batch}, int_options);

  constexpr int threads = 256;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int pixel_blocks = (total + threads - 1) / threads;
  count_roots_kernel<<<pixel_blocks, threads, 0, stream>>>(
      parent.data_ptr<int>(), component_counts.data_ptr<int>(), total);
  int batch_blocks = (batch + threads - 1) / threads;
  select_top_components_kernel<<<batch_blocks, threads, 0, stream>>>(
      component_counts.data_ptr<int>(), roots.data_ptr<int>(),
      sizes.data_ptr<int>(), raw_counts.data_ptr<int>(), plane, batch, capacity);
  component_moments_kernel<<<total_slots, threads, 0, stream>>>(
      parent.data_ptr<int>(), roots.data_ptr<int>(), angles.data_ptr<float>(),
      width, height, capacity, total_slots);
  component_extents_kernel<<<total_slots, threads, 0, stream>>>(
      parent.data_ptr<int>(), roots.data_ptr<int>(), angles.data_ptr<float>(),
      angle_offsets.data_ptr<float>(), selected_angles.data_ptr<float>(),
      extents.data_ptr<float>(), width, height, capacity, total_slots,
      angle_count);
  component_extents_kernel<<<total_slots, threads, 0, stream>>>(
      parent.data_ptr<int>(), roots.data_ptr<int>(),
      selected_angles.data_ptr<float>(), angle_refine_offsets.data_ptr<float>(),
      refined_angles.data_ptr<float>(), extents.data_ptr<float>(), width, height,
      capacity, total_slots, refine_angle_count);
  score_and_box_kernel<<<total_slots, threads, 0, stream>>>(
      probability.data_ptr<float>(), roots.data_ptr<int>(),
      sizes.data_ptr<int>(), refined_angles.data_ptr<float>(),
      extents.data_ptr<float>(), boxes.data_ptr<float>(),
      scores.data_ptr<float>(), valid.data_ptr<bool>(),
      output_counts.data_ptr<int>(), width, height, capacity, total_slots,
      static_cast<float>(box_threshold), static_cast<float>(unclip_ratio),
      static_cast<float>(min_size));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {parent, boxes, scores, valid, output_counts, raw_counts};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "connected_components", &connected_components,
      "Device-resident 8-connected components (CUDA)");
  module.def(
      "decode_db", &decode_db,
      "Device-resident fixed-capacity DB decoder (CUDA)");
}
