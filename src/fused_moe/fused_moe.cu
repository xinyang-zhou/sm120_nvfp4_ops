#include "sm120_nvfp4/fused_moe.hpp"
#include "sm120_nvfp4/grouped_gemm.hpp"

namespace sm120_nvfp4 {
namespace fused_moe {

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
    cudaStream_t stream) {
  int total_assignments = num_tokens * num_topk;
  int num_seq_per_group_avg = total_assignments / num_expert_total;

  count_and_gather_nvfp4_async(
      gate_up_input_ptr, gate_up_input_scale_ptr, input_ptr,
      input_scale_ptr, topk_ids_ptr, topk_pos_ptr, seqlens_ptr,
      cu_seqlens_ptr, num_tokens, hidden_size, num_topk,
      num_expert_local, rank_ep, input_scale_m_pad,
      grouped_scale_m_pad, stream);

  grouped_gemm::group_gemm_nvfp4_async(
      gate_up_output_ptr, gate_up_input_ptr, gate_up_weight_ptr,
      seqlens_ptr, cu_seqlens_ptr, gate_up_input_scale_ptr,
      gate_up_weight_scale_ptr, gate_up_tmas_ptr, tiles_ptr,
      cu_tiles_ptr, num_expert_local, total_assignments,
      intermediate_size * 2, hidden_size, grouped_scale_m_pad,
      num_seq_per_group_avg, true, stream);

  act_mul_and_quant_nvfp4_async(
      down_input_ptr, down_input_scale_ptr, gate_up_output_ptr,
      seqlens_ptr, cu_seqlens_ptr, num_expert_local,
      grouped_scale_m_pad, intermediate_size, stream);

  grouped_gemm::group_gemm_nvfp4_async(
      down_output_ptr, down_input_ptr, down_weight_ptr,
      seqlens_ptr, cu_seqlens_ptr, down_input_scale_ptr,
      down_weight_scale_ptr, down_tmas_ptr, tiles_ptr,
      cu_tiles_ptr, num_expert_local, total_assignments,
      hidden_size, intermediate_size, grouped_scale_m_pad,
      num_seq_per_group_avg, true, stream);

  reduce_nvfp4_async(
      output_ptr, down_output_ptr, topk_pos_ptr, topk_scale_ptr,
      shared_output_ptr, num_tokens, hidden_size, num_topk, stream);
}

}  // namespace fused_moe
}  // namespace sm120_nvfp4
