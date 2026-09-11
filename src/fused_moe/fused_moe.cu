#include "sm120_nvfp4/fused_moe.hpp"
#include "sm120_nvfp4/grouped_gemm.hpp"

#include <cstdint>
#include <stdexcept>

namespace sm120_nvfp4 {
namespace fused_moe {

namespace {

constexpr std::size_t kWorkspaceAlignment = 256;

std::size_t align_workspace(std::size_t value) {
  return (value + kWorkspaceAlignment - 1) /
         kWorkspaceAlignment * kWorkspaceAlignment;
}

std::size_t scale_k_padded_bytes(int k) {
  return static_cast<std::size_t>(((k + 15) / 16 + 3) / 4 * 4);
}

}  // namespace

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

std::size_t expert_moe_nvfp4_workspace_size(
    int total_rows, int hidden_size, int intermediate_size,
    int num_expert_local, int grouped_scale_m_pad) {
  (void)hidden_size;
  std::size_t total = 0;
  auto reserve = [&total](std::size_t bytes) {
    total = align_workspace(total);
    total += bytes;
  };
  reserve(static_cast<std::size_t>(total_rows) * intermediate_size * 2 *
          sizeof(std::uint16_t));
  reserve(static_cast<std::size_t>(total_rows) * intermediate_size / 2);
  reserve(static_cast<std::size_t>(num_expert_local) * grouped_scale_m_pad *
          scale_k_padded_bytes(intermediate_size));
  reserve(static_cast<std::size_t>(num_expert_local * 3 + 2) * 128);
  reserve(static_cast<std::size_t>(num_expert_local * 3 + 2) * 128);
  reserve(static_cast<std::size_t>(num_expert_local) * sizeof(int));
  reserve(static_cast<std::size_t>(num_expert_local + 1) * sizeof(int));
  return align_workspace(total);
}

void expert_moe_nvfp4_async(
    void *output_ptr, const void *input_ptr, const void *input_scale_ptr,
    const void *gate_up_weight_ptr, const void *gate_up_weight_scale_ptr,
    const void *down_weight_ptr, const void *down_weight_scale_ptr,
    const void *seqlens_ptr, const void *cu_seqlens_ptr,
    void *workspace_ptr, std::size_t workspace_bytes,
    int total_rows, int hidden_size, int intermediate_size,
    int num_expert_local, int grouped_scale_m_pad, cudaStream_t stream) {
  const std::size_t required = expert_moe_nvfp4_workspace_size(
      total_rows, hidden_size, intermediate_size, num_expert_local,
      grouped_scale_m_pad);
  if (workspace_ptr == nullptr || workspace_bytes < required) {
    throw std::invalid_argument("expert_moe_nvfp4 workspace is null or too small");
  }

  auto *base = static_cast<std::uint8_t *>(workspace_ptr);
  std::size_t offset = 0;
  auto take = [&](std::size_t bytes) -> void * {
    offset = align_workspace(offset);
    void *result = base + offset;
    offset += bytes;
    return result;
  };

  void *gate_up_output = take(
      static_cast<std::size_t>(total_rows) * intermediate_size * 2 *
      sizeof(std::uint16_t));
  void *down_input = take(
      static_cast<std::size_t>(total_rows) * intermediate_size / 2);
  void *down_input_scale = take(
      static_cast<std::size_t>(num_expert_local) * grouped_scale_m_pad *
      scale_k_padded_bytes(intermediate_size));
  void *gate_up_tmas = take(
      static_cast<std::size_t>(num_expert_local * 3 + 2) * 128);
  void *down_tmas = take(
      static_cast<std::size_t>(num_expert_local * 3 + 2) * 128);
  void *tiles = take(
      static_cast<std::size_t>(num_expert_local) * sizeof(int));
  void *cu_tiles = take(
      static_cast<std::size_t>(num_expert_local + 1) * sizeof(int));

  int num_seq_per_group_avg = total_rows / num_expert_local;
  grouped_gemm::group_gemm_nvfp4_async(
      gate_up_output, input_ptr, gate_up_weight_ptr,
      seqlens_ptr, cu_seqlens_ptr, input_scale_ptr,
      gate_up_weight_scale_ptr, gate_up_tmas, tiles, cu_tiles,
      num_expert_local, total_rows, intermediate_size * 2, hidden_size,
      grouped_scale_m_pad, num_seq_per_group_avg, true, stream);

  act_mul_and_quant_nvfp4_async(
      down_input, down_input_scale, gate_up_output,
      seqlens_ptr, cu_seqlens_ptr, num_expert_local,
      grouped_scale_m_pad, intermediate_size, stream);

  grouped_gemm::group_gemm_nvfp4_async(
      output_ptr, down_input, down_weight_ptr,
      seqlens_ptr, cu_seqlens_ptr, down_input_scale,
      down_weight_scale_ptr, down_tmas, tiles, cu_tiles,
      num_expert_local, total_rows, hidden_size, intermediate_size,
      grouped_scale_m_pad, num_seq_per_group_avg, true, stream);
}

}  // namespace fused_moe
}  // namespace sm120_nvfp4
