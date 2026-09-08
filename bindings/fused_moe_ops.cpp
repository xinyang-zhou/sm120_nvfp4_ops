#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cstdint>
#include <limits>
#include <optional>

#include "sm120_nvfp4/fused_moe.hpp"

namespace sm120_nvfp4 {
namespace fused_moe {
namespace {

constexpr int64_t kSFVectorSize = 16;

int64_t div_up(int64_t value, int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

int64_t scale_k_padded_host(int64_t k) {
  return div_up(div_up(k, kSFVectorSize), 4) * 4;
}

int64_t align_up_128_host(int64_t value) {
  return div_up(value, 128) * 128;
}

bool is_packed_e2m1(const torch::Tensor &tensor) {
  return tensor.scalar_type() == at::kFloat4_e2m1fn_x2 ||
         tensor.scalar_type() == torch::kUInt8;
}

void check_cuda_contiguous_on_device(
    const torch::Tensor &tensor, int device, const char *name) {
  TORCH_CHECK(tensor.device().is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.get_device() == device,
              name, " must be on the same CUDA device as input");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor fuse_moe_nvfp4_entry(
    const torch::Tensor &input, const torch::Tensor &input_scale,
    const torch::Tensor &gate_up_weight,
    const torch::Tensor &gate_up_weight_scale,
    const torch::Tensor &down_weight,
    const torch::Tensor &down_weight_scale,
    const torch::Tensor &topk_ids, const torch::Tensor &topk_scale,
    std::optional<torch::Tensor> shared_output,
    int64_t rank_ep, int64_t num_expert_total,
    std::optional<torch::Tensor> output) {
  TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
  int device = input.get_device();
  c10::cuda::CUDAGuard device_guard(device);

  check_cuda_contiguous_on_device(input, device, "input");
  check_cuda_contiguous_on_device(input_scale, device, "input_scale");
  check_cuda_contiguous_on_device(gate_up_weight, device, "gate_up_weight");
  check_cuda_contiguous_on_device(gate_up_weight_scale, device,
                                  "gate_up_weight_scale");
  check_cuda_contiguous_on_device(down_weight, device, "down_weight");
  check_cuda_contiguous_on_device(down_weight_scale, device,
                                  "down_weight_scale");
  check_cuda_contiguous_on_device(topk_ids, device, "topk_ids");
  check_cuda_contiguous_on_device(topk_scale, device, "topk_scale");

  TORCH_CHECK(is_packed_e2m1(input) && is_packed_e2m1(gate_up_weight) &&
                  is_packed_e2m1(down_weight),
              "input, gate_up_weight and down_weight must be "
              "float4_e2m1fn_x2 or packed uint8");
  TORCH_CHECK(input.scalar_type() == gate_up_weight.scalar_type() &&
                  input.scalar_type() == down_weight.scalar_type(),
              "all packed E2M1 tensors must use the same storage dtype");
  TORCH_CHECK(input_scale.scalar_type() == torch::kUInt8 &&
                  gate_up_weight_scale.scalar_type() == torch::kUInt8 &&
                  down_weight_scale.scalar_type() == torch::kUInt8,
              "all NVFP4 scale tensors must contain raw UE4M3 bytes (uint8)");
  TORCH_CHECK(topk_ids.scalar_type() == torch::kInt32,
              "topk_ids must be int32");
  TORCH_CHECK(topk_scale.scalar_type() == torch::kFloat32,
              "topk_scale must be float32");

  TORCH_CHECK(input.dim() == 2,
              "input must have packed shape [num_tokens, hidden_size/2]");
  TORCH_CHECK(gate_up_weight.dim() == 3,
              "gate_up_weight must have packed shape "
              "[num_expert_local, 2*intermediate_size, hidden_size/2]");
  TORCH_CHECK(down_weight.dim() == 3,
              "down_weight must have packed shape "
              "[num_expert_local, hidden_size, intermediate_size/2]");
  TORCH_CHECK(topk_ids.dim() == 2 && topk_scale.dim() == 2,
              "topk_ids and topk_scale must be two-dimensional");
  TORCH_CHECK(topk_ids.sizes() == topk_scale.sizes(),
              "topk_ids and topk_scale must have the same shape");
  TORCH_CHECK(topk_ids.size(0) == input.size(0),
              "topk routing tensors and input must share num_tokens");

  constexpr int64_t kIntMax = std::numeric_limits<int>::max();
  TORCH_CHECK(input.size(0) > 0 && input.size(0) <= kIntMax,
              "num_tokens must be in the positive int32 range");
  TORCH_CHECK(input.size(1) > 0 && input.size(1) <= kIntMax / 2,
              "packed hidden_size exceeds the supported int32 range");
  TORCH_CHECK(topk_ids.size(1) > 0 && topk_ids.size(1) <= 128,
              "num_topk must be in [1, 128]");

  int64_t num_tokens = input.size(0);
  int64_t hidden_size = input.size(1) * 2;
  int64_t num_expert_local = gate_up_weight.size(0);
  int64_t gate_up_size = gate_up_weight.size(1);
  int64_t num_topk = topk_ids.size(1);

  TORCH_CHECK(num_expert_local > 0 && num_expert_local <= 256,
              "num_expert_local must be in [1, 256]");
  TORCH_CHECK(gate_up_size > 0 && gate_up_size % 2 == 0,
              "gate_up_weight.size(1) must be 2 * intermediate_size");
  int64_t intermediate_size = gate_up_size / 2;
  TORCH_CHECK(hidden_size <= kIntMax && intermediate_size <= kIntMax,
              "hidden or intermediate dimension exceeds int32 range");
  TORCH_CHECK(hidden_size % 32 == 0 && intermediate_size % 32 == 0,
              "hidden_size and intermediate_size must be multiples of 32");
  TORCH_CHECK(gate_up_weight.size(2) == input.size(1),
              "input and gate_up_weight must share packed hidden_size/2");
  TORCH_CHECK(down_weight.size(0) == num_expert_local &&
                  down_weight.size(1) == hidden_size &&
                  down_weight.size(2) * 2 == intermediate_size,
              "down_weight shape must be [num_expert_local, hidden_size, "
              "intermediate_size/2]");

  TORCH_CHECK(num_expert_total > 0 && num_expert_total <= kIntMax,
              "num_expert_total must be in the positive int32 range");
  TORCH_CHECK(rank_ep >= 0 && rank_ep <= kIntMax,
              "rank_ep must be a non-negative int32 value");
  TORCH_CHECK(rank_ep * num_expert_local + num_expert_local <=
                  num_expert_total,
              "rank_ep and num_expert_local select experts outside "
              "num_expert_total");

  int64_t total_assignments = num_tokens * num_topk;
  TORCH_CHECK(total_assignments <= kIntMax,
              "num_tokens * num_topk exceeds the int32 range");
  int64_t grouped_scale_m_pad =
      std::max<int64_t>(128, align_up_128_host(total_assignments));

  int64_t input_scale_k_padded = scale_k_padded_host(hidden_size);
  TORCH_CHECK(input_scale.numel() % input_scale_k_padded == 0,
              "input_scale physical size is incompatible with hidden_size");
  int64_t input_scale_m_pad = input_scale.numel() / input_scale_k_padded;
  TORCH_CHECK(input_scale_m_pad >= num_tokens &&
                  input_scale_m_pad % 128 == 0,
              "input_scale must use a 128-row-aligned "
              "Sm1xxBlockScaledConfig<16> SFA region");

  int64_t expected_gate_scale =
      align_up_128_host(gate_up_size) * scale_k_padded_host(hidden_size);
  int64_t expected_down_scale =
      align_up_128_host(hidden_size) *
      scale_k_padded_host(intermediate_size);
  TORCH_CHECK(gate_up_weight_scale.dim() == 2 &&
                  gate_up_weight_scale.size(0) == num_expert_local &&
                  gate_up_weight_scale.size(1) == expected_gate_scale,
              "gate_up_weight_scale must have CUTLASS physical shape [",
              num_expert_local, ", ", expected_gate_scale, "]");
  TORCH_CHECK(down_weight_scale.dim() == 2 &&
                  down_weight_scale.size(0) == num_expert_local &&
                  down_weight_scale.size(1) == expected_down_scale,
              "down_weight_scale must have CUTLASS physical shape [",
              num_expert_local, ", ", expected_down_scale, "]");

  auto byte_options = input.options().dtype(torch::kUInt8);
  auto int_options = input.options().dtype(torch::kInt32);
  auto half_options = input.options().dtype(torch::kFloat16);

  torch::Tensor result;
  if (output.has_value()) {
    result = output.value();
    check_cuda_contiguous_on_device(result, device, "output");
    TORCH_CHECK(result.scalar_type() == torch::kFloat16 &&
                    result.sizes() ==
                        torch::IntArrayRef({num_tokens, hidden_size}),
                "output must be contiguous FP16 [num_tokens, hidden_size]");
  } else {
    result = torch::empty({num_tokens, hidden_size}, half_options);
  }

  const void *shared_output_ptr = nullptr;
  if (shared_output.has_value()) {
    const auto &shared = shared_output.value();
    check_cuda_contiguous_on_device(shared, device, "shared_output");
    TORCH_CHECK(shared.scalar_type() == torch::kFloat16 &&
                    shared.sizes() ==
                        torch::IntArrayRef({num_tokens, hidden_size}),
                "shared_output must be contiguous FP16 "
                "[num_tokens, hidden_size]");
    shared_output_ptr = shared.const_data_ptr();
  }

  int64_t gate_input_scale_group_elements =
      grouped_scale_m_pad * scale_k_padded_host(hidden_size);
  int64_t down_input_scale_group_elements =
      grouped_scale_m_pad * scale_k_padded_host(intermediate_size);

  torch::Tensor gate_up_input = torch::empty(
      {total_assignments, hidden_size / 2}, byte_options);
  torch::Tensor gate_up_input_scale = torch::empty(
      {num_expert_local, gate_input_scale_group_elements}, byte_options);
  torch::Tensor gate_up_output = torch::empty(
      {total_assignments, gate_up_size}, half_options);
  torch::Tensor down_input = torch::empty(
      {total_assignments, intermediate_size / 2}, byte_options);
  torch::Tensor down_input_scale = torch::empty(
      {num_expert_local, down_input_scale_group_elements}, byte_options);
  torch::Tensor down_output = torch::empty(
      {total_assignments, hidden_size}, half_options);

  torch::Tensor topk_pos = torch::empty(
      {num_tokens, num_topk}, int_options);
  torch::Tensor seqlens = torch::empty({num_expert_local}, int_options);
  torch::Tensor cu_seqlens = torch::empty({num_expert_local + 1}, int_options);
  torch::Tensor tiles = torch::empty({num_expert_local}, int_options);
  torch::Tensor cu_tiles = torch::empty({num_expert_local + 1}, int_options);
  torch::Tensor gate_up_tmas = torch::empty(
      {num_expert_local * 3 + 2, 128}, byte_options);
  torch::Tensor down_tmas = torch::empty(
      {num_expert_local * 3 + 2, 128}, byte_options);

  auto stream = at::cuda::getCurrentCUDAStream(device);
  fuse_moe_nvfp4_async(
      result.mutable_data_ptr(), input.const_data_ptr(),
      input_scale.const_data_ptr(), gate_up_input.mutable_data_ptr(),
      gate_up_input_scale.mutable_data_ptr(),
      gate_up_output.mutable_data_ptr(), gate_up_weight.const_data_ptr(),
      gate_up_weight_scale.const_data_ptr(), gate_up_tmas.mutable_data_ptr(),
      down_input.mutable_data_ptr(), down_input_scale.mutable_data_ptr(),
      down_output.mutable_data_ptr(), down_weight.const_data_ptr(),
      down_weight_scale.const_data_ptr(), down_tmas.mutable_data_ptr(),
      topk_ids.const_data_ptr(), topk_scale.const_data_ptr(),
      topk_pos.mutable_data_ptr(), seqlens.mutable_data_ptr(),
      cu_seqlens.mutable_data_ptr(), tiles.mutable_data_ptr(),
      cu_tiles.mutable_data_ptr(), shared_output_ptr,
      static_cast<int>(num_tokens), static_cast<int>(hidden_size),
      static_cast<int>(intermediate_size), static_cast<int>(num_topk),
      static_cast<int>(num_expert_total), static_cast<int>(num_expert_local),
      static_cast<int>(rank_ep), static_cast<int>(input_scale_m_pad),
      static_cast<int>(grouped_scale_m_pad), stream);

  cudaError_t status = cudaPeekAtLastError();
  TORCH_CHECK(status == cudaSuccess,
              "fuse_moe_nvfp4 launch failed: ", cudaGetErrorString(status));
  return result;
}

}  // namespace fused_moe
}  // namespace sm120_nvfp4

TORCH_LIBRARY_FRAGMENT(sm120_nvfp4, m) {
  m.def(
      "fused_moe(Tensor input, Tensor input_scale, "
      "Tensor gate_up_weight, Tensor gate_up_weight_scale, "
      "Tensor down_weight, Tensor down_weight_scale, Tensor topk_ids, "
      "Tensor topk_scale, Tensor? shared_output, int rank_ep, "
      "int num_expert_total, Tensor? output) -> Tensor");
  m.impl("fused_moe", torch::kCUDA,
         &sm120_nvfp4::fused_moe::fuse_moe_nvfp4_entry);
}
