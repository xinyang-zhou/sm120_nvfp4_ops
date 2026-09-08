#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cstdint>
#include <limits>
#include <optional>

#include "sm120_nvfp4/grouped_gemm.hpp"

namespace sm120_nvfp4 {
namespace grouped_gemm {

torch::Tensor group_gemm_nvfp4_entry(
    const torch::Tensor &x, const torch::Tensor &weight,
    const torch::Tensor &seqlens, const torch::Tensor &cu_seqlens,
    const torch::Tensor &x_scale, const torch::Tensor &w_scale,
    int64_t num_seq_per_group_avg,
    std::optional<torch::Tensor> output,
    std::optional<torch::Tensor> tma_desc) {
  TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(seqlens.device().is_cuda(), "seqlens must be a CUDA tensor");
  TORCH_CHECK(cu_seqlens.device().is_cuda(), "cu_seqlens must be a CUDA tensor");
  TORCH_CHECK(x_scale.device().is_cuda(), "x_scale must be a CUDA tensor");
  TORCH_CHECK(w_scale.device().is_cuda(), "w_scale must be a CUDA tensor");

  auto device = x.get_device();
  TORCH_CHECK(weight.get_device() == device && seqlens.get_device() == device &&
                  cu_seqlens.get_device() == device && x_scale.get_device() == device &&
                  w_scale.get_device() == device,
              "all inputs must be on the same CUDA device");
  c10::cuda::CUDAGuard device_guard(device);

  bool x_is_packed_fp4 = x.scalar_type() == at::kFloat4_e2m1fn_x2 ||
                         x.scalar_type() == torch::kUInt8;
  bool w_is_packed_fp4 = weight.scalar_type() == at::kFloat4_e2m1fn_x2 ||
                         weight.scalar_type() == torch::kUInt8;
  TORCH_CHECK(x_is_packed_fp4 && w_is_packed_fp4,
              "x and weight must be float4_e2m1fn_x2 or packed uint8");
  TORCH_CHECK(x.scalar_type() == weight.scalar_type(),
              "x and weight must use the same packed storage dtype");
  TORCH_CHECK(x_scale.scalar_type() == torch::kUInt8 &&
                  w_scale.scalar_type() == torch::kUInt8,
              "x_scale and w_scale must contain raw UE4M3 bytes (uint8)");
  TORCH_CHECK(seqlens.scalar_type() == torch::kInt32 &&
                  cu_seqlens.scalar_type() == torch::kInt32,
              "seqlens and cu_seqlens must be int32");

  TORCH_CHECK(x.dim() == 2, "x must have packed shape [M, K/2]");
  TORCH_CHECK(weight.dim() == 3,
              "weight must have packed shape [num_group, N, K/2]");
  TORCH_CHECK(x_scale.dim() == 2,
              "x_scale must have physical shape [num_group, sfa_group_elements]");
  TORCH_CHECK(w_scale.dim() == 2,
              "w_scale must have physical shape [num_group, sfb_group_elements]");
  TORCH_CHECK(seqlens.dim() == 1 && cu_seqlens.dim() == 1,
              "seqlens and cu_seqlens must be one-dimensional");

  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(x_scale.is_contiguous(), "x_scale must be contiguous");
  TORCH_CHECK(w_scale.is_contiguous(), "w_scale must be contiguous");
  TORCH_CHECK(seqlens.is_contiguous(), "seqlens must be contiguous");
  TORCH_CHECK(cu_seqlens.is_contiguous(), "cu_seqlens must be contiguous");

  constexpr int64_t kIntMax = std::numeric_limits<int>::max();
  TORCH_CHECK(weight.size(0) <= kIntMax && x.size(0) <= kIntMax &&
                  weight.size(1) <= kIntMax && x.size(1) <= kIntMax / 2,
              "problem dimensions exceed the supported int32 range");

  int num_group = static_cast<int>(weight.size(0));
  int m = static_cast<int>(x.size(0));
  int n = static_cast<int>(weight.size(1));
  int packed_k = static_cast<int>(x.size(1));
  int k = packed_k * 2;

  TORCH_CHECK(num_group > 0 && num_group <= 256,
              "num_group must be in [1, 256]");
  TORCH_CHECK(m > 0 && n > 0 && k > 0,
              "M, N and logical K must all be positive");
  TORCH_CHECK(weight.size(2) == packed_k,
              "x and weight must share the same packed K/2 dimension");
  TORCH_CHECK(seqlens.size(0) == num_group,
              "seqlens length must equal num_group");
  TORCH_CHECK(cu_seqlens.size(0) == num_group + 1,
              "cu_seqlens length must equal num_group + 1");
  TORCH_CHECK(x_scale.size(0) == num_group && w_scale.size(0) == num_group,
              "scale tensors' first dimension must equal num_group");
  TORCH_CHECK(k % 32 == 0,
              "logical K must be a multiple of 32 for aligned FP4 TMA access");
  TORCH_CHECK(num_seq_per_group_avg >= 0 && num_seq_per_group_avg <= kIntMax,
              "num_seq_per_group_avg must be in the int32 range");

  constexpr int64_t kSFVectorSize = 16;
  int64_t scale_k =
      (static_cast<int64_t>(k) + kSFVectorSize - 1) / kSFVectorSize;
  int64_t scale_k_padded = (scale_k + 3) / 4 * 4;
  TORCH_CHECK(x_scale.size(1) % scale_k_padded == 0,
              "x_scale physical size is incompatible with K and SFVectorSize=16");
  int64_t m_scale_pad_64 = x_scale.size(1) / scale_k_padded;
  TORCH_CHECK(m_scale_pad_64 <= kIntMax,
              "x_scale row capacity exceeds the supported int32 range");
  int m_scale_pad = static_cast<int>(m_scale_pad_64);
  TORCH_CHECK(m_scale_pad >= 128 && m_scale_pad % 128 == 0,
              "x_scale must reserve a 128-row-aligned SFA region for every group");

  // Descriptor addresses and group tile counts are derived directly from
  // these arrays. Validate them before launching so malformed prefix sums
  // cannot turn into asynchronous out-of-bounds TMA accesses.
  auto seqlens_cpu = seqlens.cpu();
  auto cu_seqlens_cpu = cu_seqlens.cpu();
  const int32_t *seqlens_data = seqlens_cpu.const_data_ptr<int32_t>();
  const int32_t *cu_seqlens_data = cu_seqlens_cpu.const_data_ptr<int32_t>();
  TORCH_CHECK(cu_seqlens_data[0] == 0, "cu_seqlens must start at zero");
  int64_t expected_cu_seqlen = 0;
  for (int group = 0; group < num_group; ++group) {
    int32_t group_m = seqlens_data[group];
    TORCH_CHECK(group_m >= 0, "seqlens must be non-negative; group ", group,
                " has ", group_m);
    TORCH_CHECK(group_m <= m_scale_pad,
                "x_scale does not reserve enough rows for group ", group,
                "; need ", group_m, ", have ", m_scale_pad);
    TORCH_CHECK(cu_seqlens_data[group] == expected_cu_seqlen,
                "cu_seqlens is inconsistent with seqlens at group ", group);
    expected_cu_seqlen += group_m;
    TORCH_CHECK(expected_cu_seqlen <= m,
                "sum(seqlens) exceeds x.size(0) at group ", group);
    TORCH_CHECK(cu_seqlens_data[group + 1] == expected_cu_seqlen,
                "cu_seqlens is inconsistent with seqlens at group ", group);
  }
  TORCH_CHECK(expected_cu_seqlen == m,
              "sum(seqlens) and cu_seqlens[-1] must equal x.size(0)");

  int64_t n_scale_pad =
      (static_cast<int64_t>(n) + 127) / 128 * 128;
  int64_t expected_sfb_group_elements =
      n_scale_pad * scale_k_padded;
  TORCH_CHECK(w_scale.size(1) == expected_sfb_group_elements,
              "w_scale has an invalid CUTLASS block-scaled physical size; expected ",
              expected_sfb_group_elements, " bytes per group, got ", w_scale.size(1));

  auto byte_options = x.options().dtype(torch::kUInt8);
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
    TORCH_CHECK(y.device().is_cuda() && y.get_device() == device,
                "output must be on the same CUDA device as x");
    TORCH_CHECK(y.scalar_type() == torch::kFloat16,
                "output dtype must be float16");
    TORCH_CHECK(y.sizes() == torch::IntArrayRef({m, n}),
                "output shape must be [M, N]");
    TORCH_CHECK(y.is_contiguous(), "output must be contiguous");
  } else {
    y = torch::empty({m, n}, x.options().dtype(torch::kFloat16));
  }

  torch::Tensor tmas;
  if (tma_desc.has_value()) {
    tmas = tma_desc.value();
    TORCH_CHECK(tmas.device().is_cuda() && tmas.get_device() == device,
                "tma_desc must be on the same CUDA device as x");
    TORCH_CHECK(tmas.scalar_type() == torch::kUInt8 && tmas.is_contiguous(),
                "tma_desc must be a contiguous uint8 tensor");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(tmas.data_ptr()) % 128 == 0,
                "tma_desc data pointer must be 128-byte aligned");
    TORCH_CHECK(tmas.numel() >=
                    (static_cast<int64_t>(num_group) * 3 + 2) * 128,
                "tma_desc must hold three slots per group (middle slot reserved) "
                "plus W/SFB descriptors");
  } else {
    tmas = torch::empty({num_group * 3 + 2, 128}, byte_options);
  }

  torch::Tensor tiles = torch::empty(
      {num_group}, x.options().dtype(torch::kInt32));
  torch::Tensor cu_tiles = torch::empty(
      {num_group + 1}, x.options().dtype(torch::kInt32));

  auto stream = at::cuda::getCurrentCUDAStream(device);
  group_gemm_nvfp4_async(
      y.mutable_data_ptr(), x.const_data_ptr(), weight.const_data_ptr(),
      seqlens.const_data_ptr(), cu_seqlens.const_data_ptr(),
      x_scale.const_data_ptr(), w_scale.const_data_ptr(),
      tmas.mutable_data_ptr(), tiles.mutable_data_ptr(), cu_tiles.mutable_data_ptr(),
      num_group, m, n, k, m_scale_pad,
      static_cast<int>(num_seq_per_group_avg), true, stream);

  return y;
}

}  // namespace grouped_gemm
}  // namespace sm120_nvfp4

TORCH_LIBRARY_FRAGMENT(sm120_nvfp4, m) {
  m.def(
      "grouped_gemm(Tensor x, Tensor weight, Tensor seqlens, Tensor cu_seqlens, "
      "Tensor x_scale, Tensor w_scale, int num_seq_per_group_avg, Tensor? output, "
      "Tensor? tma_desc) -> Tensor");
  m.impl("grouped_gemm", torch::kCUDA,
         &sm120_nvfp4::grouped_gemm::group_gemm_nvfp4_entry);
}
