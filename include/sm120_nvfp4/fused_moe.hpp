#ifndef SM120_NVFP4_FUSED_MOE_HPP_
#define SM120_NVFP4_FUSED_MOE_HPP_

#include <cuda_runtime_api.h>

namespace sm120_nvfp4 {
namespace fused_moe {

// Count local expert assignments, build expert-major row indices, and gather
// packed E2M1 activations plus their UE4M3 scale factors.
void count_and_gather_nvfp4_async(
    void *gate_up_input_ptr, void *gate_up_input_scale_ptr,
    const void *input_ptr, const void *input_scale_ptr,
    const void *topk_ids_ptr, void *topk_pos_ptr,
    void *seqlens_ptr, void *cu_seqlens_ptr,
    int num_tokens, int hidden_size, int num_topk,
    int num_expert_local, int rank_ep,
    int input_scale_m_pad, int grouped_scale_m_pad,
    cudaStream_t stream);

// gate_up is FP16 [rows, 2 * intermediate_size]. The output is packed E2M1
// [rows, intermediate_size / 2], and output_scale uses the CUTLASS
// Sm1xxBlockScaledConfig<16> physical SFA layout independently per expert.
void act_mul_and_quant_nvfp4_async(
    void *output_ptr, void *output_scale_ptr,
    const void *gate_up_ptr, const void *seqlens_ptr,
    const void *cu_seqlens_ptr, int num_expert_local,
    int grouped_scale_m_pad, int intermediate_size,
    cudaStream_t stream);

void reduce_nvfp4_async(
    void *output_ptr, const void *down_output_ptr,
    const void *topk_pos_ptr, const void *topk_scale_ptr,
    const void *shared_output_ptr, int num_tokens,
    int hidden_size, int num_topk, cudaStream_t stream);

// End-to-end SM120 NVFP4 MoE. All E2M1 tensors are packed two logical values
// per byte. Scale pointers contain raw UE4M3 bytes.
void fuse_moe_nvfp4_async(
    void *output_ptr, const void *input_ptr, const void *input_scale_ptr,
    void *gate_up_input_ptr, void *gate_up_input_scale_ptr,
    void *gate_up_output_ptr, const void *gate_up_weight_ptr,
    const void *gate_up_weight_scale_ptr, void *gate_up_tmas_ptr,
    void *down_input_ptr, void *down_input_scale_ptr,
    void *down_output_ptr, const void *down_weight_ptr,
    const void *down_weight_scale_ptr, void *down_tmas_ptr,
    const void *topk_ids_ptr, const void *topk_scale_ptr,
    void *topk_pos_ptr, void *seqlens_ptr, void *cu_seqlens_ptr,
    void *tiles_ptr, void *cu_tiles_ptr, const void *shared_output_ptr,
    int num_tokens, int hidden_size, int intermediate_size,
    int num_topk, int num_expert_total, int num_expert_local,
    int rank_ep, int input_scale_m_pad, int grouped_scale_m_pad,
    cudaStream_t stream);

}  // namespace fused_moe
}  // namespace sm120_nvfp4

#endif  // SM120_NVFP4_FUSED_MOE_HPP_
