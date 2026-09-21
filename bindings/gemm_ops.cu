#include "sm120_nvfp4/gemm.hpp"

#include <cstdint>
#include <limits>
#include <utility>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <torch/library.h>

namespace {

enum class GemmBackend { kAuto, kCute, kCutlass };

void check_packed_cuda_tensor(
    const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  bool packed_type =
      tensor.scalar_type() == torch::kUInt8 ||
      tensor.scalar_type() == at::kFloat4_e2m1fn_x2;
  TORCH_CHECK(
      packed_type, name,
      " must use torch.uint8 or torch.float4_e2m1fn_x2 packed storage");
  TORCH_CHECK(tensor.dim() == 2, name, " must be two-dimensional");
}

void check_scale_tensor(
    const torch::Tensor& tensor, const torch::Device& device,
    std::size_t required, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.device() == device, name, " must be on the input device");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == torch::kUInt8,
              name, " must contain raw UE4M3 bytes");
  TORCH_CHECK(
      static_cast<std::size_t>(tensor.numel()) >= required,
      name, " requires at least ", required, " physical scale bytes");
}

torch::Tensor fp4_gemm_torch(
    torch::Tensor a, torch::Tensor b,
    torch::Tensor sfa, torch::Tensor sfb,
    GemmBackend backend) {
  check_packed_cuda_tensor(a, "a");
  check_packed_cuda_tensor(b, "b");
  TORCH_CHECK(a.device() == b.device(), "a and b must be on the same device");
  TORCH_CHECK(
      a.scalar_type() == b.scalar_type(),
      "a and b must use the same packed storage dtype");
  TORCH_CHECK(
      a.size(1) == b.size(1),
      "a[M,K/2] and b[N,K/2] must have the same packed K dimension");

  int64_t m64 = a.size(0);
  int64_t n64 = b.size(0);
  int64_t k64 = a.size(1) * 2;
  constexpr int64_t kIntMax = std::numeric_limits<int>::max();
  TORCH_CHECK(
      m64 <= kIntMax && n64 <= kIntMax && k64 <= kIntMax,
      "problem dimensions exceed int32 range");

  int m = static_cast<int>(m64);
  int n = static_cast<int>(n64);
  int k = static_cast<int>(k64);
  TORCH_CHECK(
      k % sm120_nvfp4::kInputAlignmentElements == 0,
      "K must be a multiple of ", sm120_nvfp4::kInputAlignmentElements);
  TORCH_CHECK(
      n % sm120_nvfp4::kOutputAlignmentElements == 0,
      "N must be a multiple of ", sm120_nvfp4::kOutputAlignmentElements);

  check_scale_tensor(
      sfa, a.device(), sm120_nvfp4::scale_a_elements(m, n, k), "sfa");
  check_scale_tensor(
      sfb, a.device(), sm120_nvfp4::scale_b_elements(m, n, k), "sfb");

  auto output = torch::empty(
      {m64, n64}, a.options().dtype(torch::kFloat16));
  if (m == 0 || n == 0) {
    return output;
  }

  c10::cuda::CUDAGuard guard(a.device());
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(a.device().index()).stream();
  sm120_nvfp4::GemmStatus status;

  if (backend == GemmBackend::kAuto) {
    std::size_t workspace_bytes =
        sm120_nvfp4::nvfp4_gemm_workspace_size_sm120(m, n, k);
    auto workspace = torch::empty(
        {static_cast<int64_t>(workspace_bytes)},
        a.options().dtype(torch::kUInt8));
    status = sm120_nvfp4::nvfp4_gemm_sm120(
        m, n, k, a.data_ptr(), b.data_ptr(), sfa.data_ptr(), sfb.data_ptr(),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        workspace_bytes == 0 ? nullptr : workspace.data_ptr(),
        workspace_bytes, stream);
  } else if (backend == GemmBackend::kCute) {
    status = sm120_nvfp4::nvfp4_cute_gemm_sm120(
        m, n, k, a.data_ptr(), b.data_ptr(), sfa.data_ptr(), sfb.data_ptr(),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()), stream);
  } else {
    std::size_t workspace_bytes =
        sm120_nvfp4::nvfp4_cutlass_gemm_workspace_size_sm120(m, n, k);
    auto workspace = torch::empty(
        {static_cast<int64_t>(workspace_bytes)},
        a.options().dtype(torch::kUInt8));
    status = sm120_nvfp4::nvfp4_cutlass_gemm_sm120(
        m, n, k, a.data_ptr(), b.data_ptr(), sfa.data_ptr(), sfb.data_ptr(),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        workspace_bytes == 0 ? nullptr : workspace.data_ptr(),
        workspace_bytes, stream);
  }

  TORCH_CHECK(
      status == sm120_nvfp4::GemmStatus::kSuccess,
      backend == GemmBackend::kAuto
          ? "Default GEMM failed: "
          : (backend == GemmBackend::kCute
                 ? "Custom CuTe GEMM failed: "
                 : "CUTLASS reference GEMM failed: "),
      sm120_nvfp4::gemm_status_string(status));
  C10_CUDA_CHECK(cudaGetLastError());
  return output;
}

torch::Tensor gemm_torch(
    torch::Tensor a, torch::Tensor b,
    torch::Tensor sfa, torch::Tensor sfb) {
  return fp4_gemm_torch(
      std::move(a), std::move(b), std::move(sfa), std::move(sfb),
      GemmBackend::kAuto);
}

torch::Tensor cute_gemm_torch(
    torch::Tensor a, torch::Tensor b,
    torch::Tensor sfa, torch::Tensor sfb) {
  return fp4_gemm_torch(
      std::move(a), std::move(b), std::move(sfa), std::move(sfb),
      GemmBackend::kCute);
}

torch::Tensor cutlass_gemm_torch(
    torch::Tensor a, torch::Tensor b,
    torch::Tensor sfa, torch::Tensor sfb) {
  return fp4_gemm_torch(
      std::move(a), std::move(b), std::move(sfa), std::move(sfb),
      GemmBackend::kCutlass);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(sm120_nvfp4, m) {
  m.def("gemm(Tensor a, Tensor b, Tensor sfa, Tensor sfb) -> Tensor");
  m.def("cute_gemm(Tensor a, Tensor b, Tensor sfa, Tensor sfb) -> Tensor");
  m.def("cutlass_gemm(Tensor a, Tensor b, Tensor sfa, Tensor sfb) -> Tensor");
  m.impl("gemm", torch::kCUDA, &gemm_torch);
  m.impl("cute_gemm", torch::kCUDA, &cute_gemm_torch);
  m.impl("cutlass_gemm", torch::kCUDA, &cutlass_gemm_torch);
}
