#include <cuda_fp4.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "cutlass/float8.h"
#include "sm120_nvfp4/fused_moe.hpp"
#include "fused_moe/scale_layout.cuh"

namespace sm120_nvfp4 {
namespace fused_moe {
namespace kernels {

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + expf(-x));
}

__global__ void act_mul_and_quant_nvfp4_kernel(
    uint8_t *output, cutlass::float_ue4m3_t *output_scale,
    const half *gate_up, const int *seqlens, const int *cu_seqlens,
    int num_expert_local, int m_scale_pad, int intermediate_size,
    int64_t total_work) {
  int num_scale_blocks = intermediate_size / kNvfp4SFVectorSize;
  int64_t group_scale_stride =
      sfa_group_elements(m_scale_pad, intermediate_size);

  for (int64_t linear =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total_work;
       linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int k_block = static_cast<int>(linear % num_scale_blocks);
    int64_t row_and_group = linear / num_scale_blocks;
    int row = static_cast<int>(row_and_group % m_scale_pad);
    int expert = static_cast<int>(row_and_group / m_scale_pad);
    if (expert >= num_expert_local || row >= seqlens[expert]) {
      continue;
    }

    int grouped_row = cu_seqlens[expert] + row;
    const half *gate_row =
        gate_up + static_cast<int64_t>(grouped_row) * intermediate_size * 2;
    const half *up_row = gate_row + intermediate_size;
    int col = k_block * kNvfp4SFVectorSize;

    float values[kNvfp4SFVectorSize];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < kNvfp4SFVectorSize; ++i) {
      float value = silu(__half2float(gate_row[col + i])) *
                    __half2float(up_row[col + i]);
      values[i] = value;
      amax = fmaxf(amax, fabsf(value));
    }

    // E2M1's largest finite magnitude is 6. Quantize the scale first and use
    // the decoded UE4M3 value for payload quantization, matching what the MMA
    // consumes. A zero block uses a zero scale and zero payload.
    cutlass::float_ue4m3_t encoded_scale(0.0f);
    float decoded_scale = 1.0f;
    if (amax > 0.0f) {
      encoded_scale = cutlass::float_ue4m3_t(amax / 6.0f);
      decoded_scale = static_cast<float>(encoded_scale);
      if (!(decoded_scale > 0.0f)) {
        // Preserve tiny non-zero blocks with UE4M3's smallest denormal
        // instead of replacing their scale with 1.0.
        encoded_scale.raw() = 1;
        decoded_scale = static_cast<float>(encoded_scale);
      }
    }

    uint8_t *output_row =
        output + static_cast<int64_t>(grouped_row) * intermediate_size / 2;
#pragma unroll
    for (int i = 0; i < kNvfp4SFVectorSize / 2; ++i) {
      float2 pair;
      pair.x = values[i * 2] / decoded_scale;
      pair.y = values[i * 2 + 1] / decoded_scale;
      __nv_fp4x2_e2m1 packed(pair);
      output_row[col / 2 + i] = packed.__x;
    }

    int64_t scale_offset =
        sfa_offset(row, k_block, m_scale_pad, intermediate_size);
    output_scale[static_cast<int64_t>(expert) * group_scale_stride +
                 scale_offset] = encoded_scale;
  }
}

}  // namespace kernels

void act_mul_and_quant_nvfp4_async(
    void *output_ptr, void *output_scale_ptr,
    const void *gate_up_ptr, const void *seqlens_ptr,
    const void *cu_seqlens_ptr, int num_expert_local,
    int grouped_scale_m_pad, int intermediate_size,
    cudaStream_t stream) {
  int64_t group_scale_stride =
      sfa_group_elements(grouped_scale_m_pad, intermediate_size);
  cudaMemsetAsync(output_scale_ptr, 0,
                  static_cast<size_t>(num_expert_local) * group_scale_stride,
                  stream);

  int64_t total_work = static_cast<int64_t>(num_expert_local) *
                       grouped_scale_m_pad *
                       (intermediate_size / kNvfp4SFVectorSize);
  constexpr int kThreads = 256;
  int64_t wanted_blocks = (total_work + kThreads - 1) / kThreads;
  int blocks = static_cast<int>(wanted_blocks > 65535 ? 65535 : wanted_blocks);
  kernels::act_mul_and_quant_nvfp4_kernel<<<blocks, kThreads, 0, stream>>>(
      static_cast<uint8_t *>(output_ptr),
      static_cast<cutlass::float_ue4m3_t *>(output_scale_ptr),
      static_cast<const half *>(gate_up_ptr),
      static_cast<const int *>(seqlens_ptr),
      static_cast<const int *>(cu_seqlens_ptr), num_expert_local,
      grouped_scale_m_pad, intermediate_size, total_work);
}

}  // namespace fused_moe
}  // namespace sm120_nvfp4
