#include <cuda_runtime.h>

#include <cstdint>

#include "sm120_nvfp4/fused_moe.hpp"
#include "fused_moe/scale_layout.cuh"

namespace sm120_nvfp4 {
namespace fused_moe {
namespace kernels {

__global__ void count_local_experts_kernel(
    const int *topk_ids, int *topk_pos, int *seqlens,
    int total_assignments, int start_expert, int end_expert) {
  int assignment = blockIdx.x * blockDim.x + threadIdx.x;
  if (assignment >= total_assignments) {
    return;
  }
  int expert = topk_ids[assignment];
  topk_pos[assignment] = -1;
  if (expert >= start_expert && expert < end_expert) {
    atomicAdd(seqlens + expert - start_expert, 1);
  }
}

__global__ void build_prefix_and_reset_kernel(
    int *seqlens, int *cu_seqlens, int num_expert_local) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  int prefix = 0;
  cu_seqlens[0] = 0;
  for (int expert = 0; expert < num_expert_local; ++expert) {
    prefix += seqlens[expert];
    cu_seqlens[expert + 1] = prefix;
    seqlens[expert] = 0;
  }
}

__global__ void gather_nvfp4_kernel(
    uint8_t *grouped_input, uint8_t *grouped_scale,
    const uint8_t *input, const uint8_t *input_scale,
    const int *topk_ids, int *topk_pos, int *seqlens,
    const int *cu_seqlens, int total_assignments,
    int hidden_size, int num_topk, int start_expert, int end_expert,
    int input_scale_m_pad, int grouped_scale_m_pad) {
  constexpr int kWarpSize = 32;
  int lane = threadIdx.x & (kWarpSize - 1);
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) / kWarpSize;
  if (warp >= total_assignments) {
    return;
  }

  int expert_global = topk_ids[warp];
  if (expert_global < start_expert || expert_global >= end_expert) {
    return;
  }
  int expert = expert_global - start_expert;
  int position = 0;
  if (lane == 0) {
    position = atomicAdd(seqlens + expert, 1);
  }
  position = __shfl_sync(0xffffffff, position, 0);
  int grouped_row = cu_seqlens[expert] + position;
  int token = warp / num_topk;

  int packed_hidden = hidden_size / 2;
  const uint8_t *src_row = input + static_cast<int64_t>(token) * packed_hidden;
  uint8_t *dst_row =
      grouped_input + static_cast<int64_t>(grouped_row) * packed_hidden;
  for (int byte = lane; byte < packed_hidden; byte += kWarpSize) {
    dst_row[byte] = src_row[byte];
  }

  int num_scale_blocks = (hidden_size + kNvfp4SFVectorSize - 1) /
                         kNvfp4SFVectorSize;
  int64_t dst_group_stride =
      sfa_group_elements(grouped_scale_m_pad, hidden_size);
  for (int k_block = lane; k_block < num_scale_blocks;
       k_block += kWarpSize) {
    int64_t src_offset =
        sfa_offset(token, k_block, input_scale_m_pad, hidden_size);
    int64_t dst_offset =
        sfa_offset(position, k_block, grouped_scale_m_pad, hidden_size);
    grouped_scale[static_cast<int64_t>(expert) * dst_group_stride +
                  dst_offset] = input_scale[src_offset];
  }

  if (lane == 0) {
    topk_pos[warp] = grouped_row;
  }
}

}  // namespace kernels

void count_and_gather_nvfp4_async(
    void *gate_up_input_ptr, void *gate_up_input_scale_ptr,
    const void *input_ptr, const void *input_scale_ptr,
    const void *topk_ids_ptr, void *topk_pos_ptr,
    void *seqlens_ptr, void *cu_seqlens_ptr,
    int num_tokens, int hidden_size, int num_topk,
    int num_expert_local, int rank_ep,
    int input_scale_m_pad, int grouped_scale_m_pad,
    cudaStream_t stream) {
  int total_assignments = num_tokens * num_topk;
  int start_expert = rank_ep * num_expert_local;
  int end_expert = start_expert + num_expert_local;

  cudaMemsetAsync(seqlens_ptr, 0,
                  static_cast<size_t>(num_expert_local) * sizeof(int), stream);
  cudaMemsetAsync(
      gate_up_input_scale_ptr, 0,
      static_cast<size_t>(num_expert_local) *
          sfa_group_elements(grouped_scale_m_pad, hidden_size),
      stream);

  constexpr int kCountThreads = 256;
  int count_blocks =
      (total_assignments + kCountThreads - 1) / kCountThreads;
  kernels::count_local_experts_kernel<<<count_blocks, kCountThreads, 0, stream>>>(
      static_cast<const int *>(topk_ids_ptr),
      static_cast<int *>(topk_pos_ptr), static_cast<int *>(seqlens_ptr),
      total_assignments, start_expert, end_expert);

  kernels::build_prefix_and_reset_kernel<<<1, 1, 0, stream>>>(
      static_cast<int *>(seqlens_ptr), static_cast<int *>(cu_seqlens_ptr),
      num_expert_local);

  constexpr int kGatherThreads = 128;
  constexpr int kWarpsPerBlock = kGatherThreads / 32;
  int gather_blocks =
      (total_assignments + kWarpsPerBlock - 1) / kWarpsPerBlock;
  kernels::gather_nvfp4_kernel<<<gather_blocks, kGatherThreads, 0, stream>>>(
      static_cast<uint8_t *>(gate_up_input_ptr),
      static_cast<uint8_t *>(gate_up_input_scale_ptr),
      static_cast<const uint8_t *>(input_ptr),
      static_cast<const uint8_t *>(input_scale_ptr),
      static_cast<const int *>(topk_ids_ptr),
      static_cast<int *>(topk_pos_ptr), static_cast<int *>(seqlens_ptr),
      static_cast<const int *>(cu_seqlens_ptr), total_assignments,
      hidden_size, num_topk, start_expert, end_expert,
      input_scale_m_pad, grouped_scale_m_pad);
}

}  // namespace fused_moe
}  // namespace sm120_nvfp4
