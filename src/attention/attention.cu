#include "sm120_nvfp4/attention.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <cuda_fp4.h>
#include <cuda_runtime.h>

#include "cute/tensor.hpp"
#include "cutlass/float8.h"
#include "fused_moe/scale_layout.cuh"

namespace sm120_nvfp4 {
namespace attention_detail {

constexpr std::size_t kWorkspaceAlignment = 256;
constexpr int kSoftmaxThreads = 256;

constexpr std::size_t align_workspace(std::size_t value) {
  return (value + kWorkspaceAlignment - 1) /
         kWorkspaceAlignment * kWorkspaceAlignment;
}

__global__ void softmax_quantize_nvfp4_kernel(
    const float* logits, std::uint8_t* probabilities,
    cutlass::float_ue4m3_t* probability_scale,
    float* row_correction,
    int matrices, int query_length, int kv_length,
    int heads, const int* kv_lengths,
    int query_scale_rows, std::int64_t scale_stride,
    bool causal, float softmax_scale) {
  __shared__ float reduction[kSoftmaxThreads];

  const std::int64_t total_rows =
      static_cast<std::int64_t>(matrices) * query_length;
  const std::int64_t logits_matrix_stride =
      static_cast<std::int64_t>(query_length) * kv_length;
  const std::int64_t probability_matrix_stride = logits_matrix_stride / 2;

  for (std::int64_t linear_row = blockIdx.x;
       linear_row < total_rows; linear_row += gridDim.x) {
    int matrix = static_cast<int>(linear_row / query_length);
    int row = static_cast<int>(linear_row % query_length);
    int last_visible;
    if (kv_lengths != nullptr) {
      int batch = matrix / heads;
      int valid_length = kv_lengths[batch];
      valid_length = valid_length < 0 ? 0 : valid_length;
      valid_length = valid_length > kv_length ? kv_length : valid_length;
      last_visible = valid_length - 1;
    } else {
      last_visible = causal ? kv_length - query_length + row
                            : kv_length - 1;
    }
    const float* logits_row =
        logits + static_cast<std::int64_t>(matrix) * logits_matrix_stride +
        static_cast<std::int64_t>(row) * kv_length;
    std::uint8_t* probability_row =
        probabilities +
        static_cast<std::int64_t>(matrix) * probability_matrix_stride +
        static_cast<std::int64_t>(row) * kv_length / 2;

    float thread_max = -FLT_MAX;
    for (int column = threadIdx.x; column < kv_length;
         column += blockDim.x) {
      if (column <= last_visible) {
        thread_max = fmaxf(thread_max, logits_row[column] * softmax_scale);
      }
    }
    reduction[threadIdx.x] = thread_max;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
      if (threadIdx.x < offset) {
        reduction[threadIdx.x] =
            fmaxf(reduction[threadIdx.x],
                  reduction[threadIdx.x + offset]);
      }
      __syncthreads();
    }
    float row_max = reduction[0];

    float thread_sum = 0.0f;
    for (int column = threadIdx.x; column < kv_length;
         column += blockDim.x) {
      if (column <= last_visible) {
        thread_sum +=
            expf(logits_row[column] * softmax_scale - row_max);
      }
    }
    reduction[threadIdx.x] = thread_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
      if (threadIdx.x < offset) {
        reduction[threadIdx.x] += reduction[threadIdx.x + offset];
      }
      __syncthreads();
    }
    float inverse_sum = reduction[0] > 0.0f ? 1.0f / reduction[0] : 0.0f;

    float thread_quantized_sum = 0.0f;
    int scale_blocks = kv_length / kNvfp4ScaleVectorSize;
    for (int k_block = threadIdx.x; k_block < scale_blocks;
         k_block += blockDim.x) {
      int column = k_block * kNvfp4ScaleVectorSize;
      float values[kNvfp4ScaleVectorSize];
      float amax = 0.0f;
#pragma unroll
      for (int i = 0; i < kNvfp4ScaleVectorSize; ++i) {
        int current_column = column + i;
        float value = 0.0f;
        if (current_column <= last_visible) {
          value = expf(logits_row[current_column] * softmax_scale - row_max) *
                  inverse_sum;
        }
        values[i] = value;
        amax = fmaxf(amax, value);
      }

      cutlass::float_ue4m3_t encoded_scale(0.0f);
      float decoded_scale = 1.0f;
      if (amax > 0.0f) {
        encoded_scale = cutlass::float_ue4m3_t(amax / 6.0f);
        decoded_scale = static_cast<float>(encoded_scale);
        if (!(decoded_scale > 0.0f)) {
          encoded_scale.raw() = 1;
          decoded_scale = static_cast<float>(encoded_scale);
        }
      }

#pragma unroll
      for (int i = 0; i < kNvfp4ScaleVectorSize / 2; ++i) {
        float2 pair;
        pair.x = values[i * 2] / decoded_scale;
        pair.y = values[i * 2 + 1] / decoded_scale;
        __nv_fp4x2_e2m1 packed(pair);
        probability_row[column / 2 + i] = packed.__x;
        float2 reconstructed = static_cast<float2>(packed);
        thread_quantized_sum +=
            (reconstructed.x + reconstructed.y) * decoded_scale;
      }

      std::int64_t scale_offset = fused_moe::sfa_offset(
          row, k_block, query_scale_rows, kv_length);
      probability_scale[static_cast<std::int64_t>(matrix) * scale_stride +
                        scale_offset] = encoded_scale;
    }
    reduction[threadIdx.x] = thread_quantized_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
      if (threadIdx.x < offset) {
        reduction[threadIdx.x] += reduction[threadIdx.x + offset];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      row_correction[linear_row] =
          reduction[0] > 0.0f ? 1.0f / reduction[0] : 0.0f;
    }
    __syncthreads();
  }
}

__global__ void apply_row_correction_kernel(
    half* output, const float* row_correction,
    std::int64_t output_elements, int value_dim) {
  for (std::int64_t index =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < output_elements;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    std::int64_t row = index / value_dim;
    output[index] = __float2half(
        __half2float(output[index]) * row_correction[row]);
  }
}

bool valid_problem(
    int batch, int heads, int query_length, int kv_length,
    int head_dim, int value_dim, bool causal, float softmax_scale) {
  std::int64_t matrices =
      static_cast<std::int64_t>(batch) * static_cast<std::int64_t>(heads);
  return batch > 0 && heads > 0 && matrices <= std::numeric_limits<int>::max() &&
         query_length > 0 && kv_length > 0 &&
         head_dim > 0 && value_dim > 0 &&
         head_dim % kInputAlignmentElements == 0 &&
         kv_length % kInputAlignmentElements == 0 &&
         value_dim % kOutputAlignmentElements == 0 &&
         (!causal || query_length <= kv_length) &&
         std::isfinite(softmax_scale) && softmax_scale > 0.0f;
}

}  // namespace attention_detail

std::size_t nvfp4_attention_workspace_size_sm120(
    int batch, int heads, int query_length, int kv_length) {
  if (batch <= 0 || heads <= 0 || query_length <= 0 || kv_length <= 0 ||
      kv_length % kInputAlignmentElements != 0) {
    return 0;
  }
  std::size_t matrices =
      static_cast<std::size_t>(batch) * static_cast<std::size_t>(heads);
  std::size_t matrix_elements =
      static_cast<std::size_t>(query_length) *
      static_cast<std::size_t>(kv_length);
  std::size_t offset = 0;
  offset = attention_detail::align_workspace(offset);
  offset += matrices * matrix_elements * sizeof(float);
  offset = attention_detail::align_workspace(offset);
  offset += matrices * packed_fp4_bytes(query_length, kv_length);
  offset = attention_detail::align_workspace(offset);
  offset += matrices * scale_a_elements(query_length, 1, kv_length);
  offset = attention_detail::align_workspace(offset);
  offset += matrices * static_cast<std::size_t>(query_length) * sizeof(float);
  return attention_detail::align_workspace(offset);
}

GemmStatus nvfp4_attention_prefill_sm120(
    int batch, int heads, int query_length, int kv_length,
    int head_dim, int value_dim,
    const void* query, const void* key, const void* value_transposed,
    const void* query_scale, const void* key_scale,
    const void* value_scale,
    bool causal, float softmax_scale,
    half* output,
    void* workspace, std::size_t workspace_bytes,
    cudaStream_t stream) {
  if (!attention_detail::valid_problem(
          batch, heads, query_length, kv_length, head_dim, value_dim,
          causal, softmax_scale) ||
      query == nullptr || key == nullptr || value_transposed == nullptr ||
      query_scale == nullptr || key_scale == nullptr ||
      value_scale == nullptr || output == nullptr || workspace == nullptr) {
    return GemmStatus::kInvalidArgument;
  }

  std::size_t required = nvfp4_attention_workspace_size_sm120(
      batch, heads, query_length, kv_length);
  if (required == 0 || workspace_bytes < required) {
    return GemmStatus::kInsufficientWorkspace;
  }

  std::size_t matrices =
      static_cast<std::size_t>(batch) * static_cast<std::size_t>(heads);
  std::size_t logits_elements_per_matrix =
      static_cast<std::size_t>(query_length) * kv_length;
  std::size_t logits_bytes =
      matrices * logits_elements_per_matrix * sizeof(float);
  std::size_t probability_bytes_per_matrix =
      packed_fp4_bytes(query_length, kv_length);
  std::size_t probability_scale_elements_per_matrix =
      scale_a_elements(query_length, value_dim, kv_length);

  auto* workspace_bytes_ptr = static_cast<std::uint8_t*>(workspace);
  std::size_t offset = 0;
  offset = attention_detail::align_workspace(offset);
  float* logits = reinterpret_cast<float*>(workspace_bytes_ptr + offset);
  offset += logits_bytes;
  offset = attention_detail::align_workspace(offset);
  std::uint8_t* probabilities = workspace_bytes_ptr + offset;
  offset += matrices * probability_bytes_per_matrix;
  offset = attention_detail::align_workspace(offset);
  auto* probability_scale =
      reinterpret_cast<cutlass::float_ue4m3_t*>(workspace_bytes_ptr + offset);
  offset += matrices * probability_scale_elements_per_matrix;
  offset = attention_detail::align_workspace(offset);
  float* row_correction =
      reinterpret_cast<float*>(workspace_bytes_ptr + offset);

  const auto* query_bytes = static_cast<const std::uint8_t*>(query);
  const auto* key_bytes = static_cast<const std::uint8_t*>(key);
  const auto* value_bytes =
      static_cast<const std::uint8_t*>(value_transposed);
  const auto* query_scale_bytes =
      static_cast<const std::uint8_t*>(query_scale);
  const auto* key_scale_bytes = static_cast<const std::uint8_t*>(key_scale);
  const auto* value_scale_bytes =
      static_cast<const std::uint8_t*>(value_scale);

  std::size_t query_bytes_per_matrix =
      packed_fp4_bytes(query_length, head_dim);
  std::size_t key_bytes_per_matrix = packed_fp4_bytes(kv_length, head_dim);
  std::size_t value_bytes_per_matrix =
      packed_fp4_bytes(value_dim, kv_length);
  std::size_t query_scale_elements_per_matrix =
      scale_a_elements(query_length, kv_length, head_dim);
  std::size_t key_scale_elements_per_matrix =
      scale_b_elements(query_length, kv_length, head_dim);
  std::size_t value_scale_elements_per_matrix =
      scale_b_elements(query_length, value_dim, kv_length);

  for (std::size_t matrix = 0; matrix < matrices; ++matrix) {
    GemmStatus status = nvfp4_cute_gemm_f32_sm120(
        query_length, kv_length, head_dim,
        query_bytes + matrix * query_bytes_per_matrix,
        key_bytes + matrix * key_bytes_per_matrix,
        query_scale_bytes + matrix * query_scale_elements_per_matrix,
        key_scale_bytes + matrix * key_scale_elements_per_matrix,
        logits + matrix * logits_elements_per_matrix, stream);
    if (status != GemmStatus::kSuccess) {
      return status;
    }
  }

  cudaError_t error = cudaMemsetAsync(
      probability_scale, 0,
      matrices * probability_scale_elements_per_matrix, stream);
  if (error != cudaSuccess) {
    return GemmStatus::kCudaError;
  }

  std::int64_t total_rows =
      static_cast<std::int64_t>(matrices) * query_length;
  int blocks = static_cast<int>(std::min<std::int64_t>(total_rows, 65535));
  attention_detail::softmax_quantize_nvfp4_kernel<<<
      blocks, attention_detail::kSoftmaxThreads, 0, stream>>>(
      logits, probabilities, probability_scale, row_correction,
      static_cast<int>(matrices), query_length, kv_length,
      heads, /*kv_lengths=*/nullptr,
      align_up(query_length, kScaleMNAlignment),
      static_cast<std::int64_t>(probability_scale_elements_per_matrix),
      causal, softmax_scale);
  if (cudaPeekAtLastError() != cudaSuccess) {
    return GemmStatus::kCudaError;
  }

  std::size_t output_elements_per_matrix =
      static_cast<std::size_t>(query_length) * value_dim;
  for (std::size_t matrix = 0; matrix < matrices; ++matrix) {
    GemmStatus status = nvfp4_cute_gemm_sm120(
        query_length, value_dim, kv_length,
        probabilities + matrix * probability_bytes_per_matrix,
        value_bytes + matrix * value_bytes_per_matrix,
        probability_scale + matrix * probability_scale_elements_per_matrix,
        value_scale_bytes + matrix * value_scale_elements_per_matrix,
        output + matrix * output_elements_per_matrix, stream);
    if (status != GemmStatus::kSuccess) {
      return status;
    }
  }

  std::int64_t output_elements =
      static_cast<std::int64_t>(matrices) * query_length * value_dim;
  int correction_blocks = static_cast<int>(std::min<std::int64_t>(
      (output_elements + attention_detail::kSoftmaxThreads - 1) /
          attention_detail::kSoftmaxThreads,
      65535));
  attention_detail::apply_row_correction_kernel<<<
      correction_blocks, attention_detail::kSoftmaxThreads, 0, stream>>>(
      output, row_correction, output_elements, value_dim);
  if (cudaPeekAtLastError() != cudaSuccess) {
    return GemmStatus::kCudaError;
  }
  return GemmStatus::kSuccess;
}

}  // namespace sm120_nvfp4
