#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "sm120_nvfp4/fused_moe.hpp"

namespace sm120_nvfp4 {
namespace fused_moe {
namespace kernels {

__global__ void reduce_nvfp4_kernel(
    half *output, const half *down_output, const int *topk_pos,
    const float *topk_scale, const half *shared_output,
    int num_tokens, int hidden_size, int num_topk, int64_t numel) {
  for (int64_t linear =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < numel;
       linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int token = static_cast<int>(linear / hidden_size);
    int col = static_cast<int>(linear -
                               static_cast<int64_t>(token) * hidden_size);
    float value = shared_output ? __half2float(shared_output[linear]) : 0.0f;
    int route_base = token * num_topk;
    for (int route = 0; route < num_topk; ++route) {
      int row = topk_pos[route_base + route];
      if (row >= 0) {
        value += __half2float(
                     down_output[static_cast<int64_t>(row) * hidden_size + col]) *
                 topk_scale[route_base + route];
      }
    }
    output[linear] = __float2half_rn(value);
  }
}

}  // namespace kernels

void reduce_nvfp4_async(
    void *output_ptr, const void *down_output_ptr,
    const void *topk_pos_ptr, const void *topk_scale_ptr,
    const void *shared_output_ptr, int num_tokens,
    int hidden_size, int num_topk, cudaStream_t stream) {
  int64_t numel = static_cast<int64_t>(num_tokens) * hidden_size;
  constexpr int kThreads = 256;
  int64_t wanted_blocks = (numel + kThreads - 1) / kThreads;
  int blocks = static_cast<int>(wanted_blocks > 65535 ? 65535 : wanted_blocks);
  kernels::reduce_nvfp4_kernel<<<blocks, kThreads, 0, stream>>>(
      static_cast<half *>(output_ptr),
      static_cast<const half *>(down_output_ptr),
      static_cast<const int *>(topk_pos_ptr),
      static_cast<const float *>(topk_scale_ptr),
      static_cast<const half *>(shared_output_ptr), num_tokens,
      hidden_size, num_topk, numel);
}

}  // namespace fused_moe
}  // namespace sm120_nvfp4
