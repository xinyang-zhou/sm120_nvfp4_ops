#include "sm120_nvfp4/attention.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <torch/library.h>

namespace {

bool is_packed_e2m1(const torch::Tensor& tensor) {
  return tensor.scalar_type() == at::kFloat4_e2m1fn_x2 ||
         tensor.scalar_type() == torch::kUInt8;
}

void check_cuda_contiguous_on_device(
    const torch::Tensor& tensor, int device, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.get_device() == device,
              name, " must be on the same CUDA device as query");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

torch::Tensor attention_prefill_torch(
    const torch::Tensor& query, const torch::Tensor& key,
    const torch::Tensor& value_transposed,
    const torch::Tensor& query_scale, const torch::Tensor& key_scale,
    const torch::Tensor& value_scale,
    bool causal, double softmax_scale,
    std::optional<torch::Tensor> output,
    std::optional<torch::Tensor> workspace) {
  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
  int device = query.get_device();
  c10::cuda::CUDAGuard guard(device);

  check_cuda_contiguous_on_device(query, device, "query");
  check_cuda_contiguous_on_device(key, device, "key");
  check_cuda_contiguous_on_device(
      value_transposed, device, "value_transposed");
  check_cuda_contiguous_on_device(query_scale, device, "query_scale");
  check_cuda_contiguous_on_device(key_scale, device, "key_scale");
  check_cuda_contiguous_on_device(value_scale, device, "value_scale");
  TORCH_CHECK(
      is_packed_e2m1(query) && is_packed_e2m1(key) &&
          is_packed_e2m1(value_transposed),
      "query, key and value_transposed must use packed uint8 or "
      "float4_e2m1fn_x2 storage");
  TORCH_CHECK(
      query.scalar_type() == key.scalar_type() &&
          query.scalar_type() == value_transposed.scalar_type(),
      "all packed operands must use the same storage dtype");
  TORCH_CHECK(
      query_scale.scalar_type() == torch::kUInt8 &&
          key_scale.scalar_type() == torch::kUInt8 &&
          value_scale.scalar_type() == torch::kUInt8,
      "query_scale, key_scale and value_scale must contain raw UE4M3 bytes");

  TORCH_CHECK(query.dim() == 4,
              "query must have packed shape [B,H,M,D/2]");
  TORCH_CHECK(key.dim() == 4,
              "key must have packed shape [B,H,N,D/2]");
  TORCH_CHECK(value_transposed.dim() == 4,
              "value_transposed must have packed shape [B,H,Dv,N/2]");
  TORCH_CHECK(query.size(0) == key.size(0) &&
                  query.size(0) == value_transposed.size(0) &&
                  query.size(1) == key.size(1) &&
                  query.size(1) == value_transposed.size(1),
              "query, key and value_transposed must share B and H");
  TORCH_CHECK(query.size(3) == key.size(3),
              "query and key must share packed D/2");
  TORCH_CHECK(value_transposed.size(3) * 2 == key.size(2),
              "value_transposed's packed N/2 must match key's N");

  constexpr std::int64_t kIntMax = std::numeric_limits<int>::max();
  for (std::int64_t dimension : query.sizes()) {
    TORCH_CHECK(dimension > 0 && dimension <= kIntMax,
                "query dimensions must be in the positive int32 range");
  }
  TORCH_CHECK(key.size(2) > 0 && key.size(2) <= kIntMax &&
                  value_transposed.size(2) > 0 &&
                  value_transposed.size(2) <= kIntMax &&
                  query.size(3) <= kIntMax / 2,
              "attention dimensions exceed the supported int32 range");

  int batch = static_cast<int>(query.size(0));
  int heads = static_cast<int>(query.size(1));
  int query_length = static_cast<int>(query.size(2));
  int kv_length = static_cast<int>(key.size(2));
  int head_dim = static_cast<int>(query.size(3) * 2);
  int value_dim = static_cast<int>(value_transposed.size(2));
  TORCH_CHECK(
      static_cast<std::int64_t>(batch) * heads <= kIntMax,
      "batch * heads exceeds the supported int32 range");
  TORCH_CHECK(head_dim % sm120_nvfp4::kInputAlignmentElements == 0,
              "logical head_dim must be a multiple of 32");
  TORCH_CHECK(kv_length % sm120_nvfp4::kInputAlignmentElements == 0,
              "kv_length must be a multiple of 32 for P @ V");
  TORCH_CHECK(value_dim % sm120_nvfp4::kOutputAlignmentElements == 0,
              "value_dim must be a multiple of 8");
  TORCH_CHECK(!causal || query_length <= kv_length,
              "causal suffix alignment requires query_length <= kv_length");
  TORCH_CHECK(std::isfinite(softmax_scale) && softmax_scale > 0.0,
              "softmax_scale must be finite and positive");

  std::size_t matrices =
      static_cast<std::size_t>(batch) * static_cast<std::size_t>(heads);
  std::size_t required_query_scale = matrices *
      sm120_nvfp4::scale_a_elements(query_length, kv_length, head_dim);
  std::size_t required_key_scale = matrices *
      sm120_nvfp4::scale_b_elements(query_length, kv_length, head_dim);
  std::size_t required_value_scale = matrices *
      sm120_nvfp4::scale_b_elements(query_length, value_dim, kv_length);
  TORCH_CHECK(static_cast<std::size_t>(query_scale.numel()) >=
                  required_query_scale,
              "query_scale is too small; need ", required_query_scale,
              " physical UE4M3 bytes");
  TORCH_CHECK(static_cast<std::size_t>(key_scale.numel()) >=
                  required_key_scale,
              "key_scale is too small; need ", required_key_scale,
              " physical UE4M3 bytes");
  TORCH_CHECK(static_cast<std::size_t>(value_scale.numel()) >=
                  required_value_scale,
              "value_scale is too small; need ", required_value_scale,
              " physical UE4M3 bytes");

  torch::Tensor result;
  if (output.has_value()) {
    result = output.value();
    check_cuda_contiguous_on_device(result, device, "output");
    TORCH_CHECK(
        result.scalar_type() == torch::kFloat16 &&
            result.sizes() ==
                torch::IntArrayRef({batch, heads, query_length, value_dim}),
        "output must be contiguous FP16 [B,H,M,Dv]");
  } else {
    result = torch::empty(
        {batch, heads, query_length, value_dim},
        query.options().dtype(torch::kFloat16));
  }

  std::size_t required_workspace =
      sm120_nvfp4::nvfp4_attention_workspace_size_sm120(
          batch, heads, query_length, kv_length);
  torch::Tensor scratch;
  if (workspace.has_value()) {
    scratch = workspace.value();
    check_cuda_contiguous_on_device(scratch, device, "workspace");
    TORCH_CHECK(scratch.scalar_type() == torch::kUInt8,
                "workspace must be uint8");
    TORCH_CHECK(static_cast<std::size_t>(scratch.numel()) >=
                    required_workspace,
                "workspace is too small; need ", required_workspace,
                " bytes");
  } else {
    scratch = torch::empty(
        {static_cast<std::int64_t>(required_workspace)},
        query.options().dtype(torch::kUInt8));
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
  sm120_nvfp4::GemmStatus status =
      sm120_nvfp4::nvfp4_attention_prefill_sm120(
          batch, heads, query_length, kv_length, head_dim, value_dim,
          query.const_data_ptr(), key.const_data_ptr(),
          value_transposed.const_data_ptr(), query_scale.const_data_ptr(),
          key_scale.const_data_ptr(), value_scale.const_data_ptr(), causal,
          static_cast<float>(softmax_scale),
          reinterpret_cast<half*>(result.mutable_data_ptr<at::Half>()),
          scratch.mutable_data_ptr(), required_workspace, stream);
  TORCH_CHECK(status == sm120_nvfp4::GemmStatus::kSuccess,
              "SM120 NVFP4 attention failed: ",
              sm120_nvfp4::gemm_status_string(status));
  C10_CUDA_CHECK(cudaGetLastError());
  return result;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(sm120_nvfp4, m) {
  m.def(
      "attention_prefill(Tensor query, Tensor key, Tensor value_transposed, "
      "Tensor query_scale, Tensor key_scale, Tensor value_scale, bool causal, "
      "float softmax_scale, Tensor? output, Tensor? workspace) -> Tensor");
  m.impl("attention_prefill", torch::kCUDA, &attention_prefill_torch);
}
