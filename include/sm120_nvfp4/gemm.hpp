#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "cutlass/float_subbyte.h"

namespace sm120_nvfp4 {

// NVFP4 stores two E2M1 payloads per byte and uses one UE4M3 scale for every
// 16 consecutive K values. Scale buffers use CUTLASS Sm1xxBlockScaledConfig<16>
// physical layouts, not a flat logical [M, K/16] layout.
using Nvfp4Element = cutlass::float_e2m1_t;
using Nvfp4Scale = cutlass::float_ue4m3_t;
using PackedStorage = std::uint8_t;
using OutputElement = half;

constexpr int kNvfp4ScaleVectorSize = 16;
constexpr int kScaleMNAlignment = 128;
constexpr int kInputAlignmentElements = 32;
constexpr int kOutputAlignmentElements = 8;

inline constexpr int ceil_div(int x, int y) {
  return (x + y - 1) / y;
}

inline constexpr int align_up(int x, int alignment) {
  return ceil_div(x, alignment) * alignment;
}

// Bytes for a packed row-major E2M1 matrix [rows, logical_cols].
inline constexpr std::size_t packed_fp4_bytes(int rows, int logical_cols) {
  return static_cast<std::size_t>(rows) *
         static_cast<std::size_t>(ceil_div(logical_cols, 2));
}

// CUTLASS pads the number of scale vectors in K to a multiple of four.
inline constexpr int scale_k_padded(int k) {
  return align_up(ceil_div(k, kNvfp4ScaleVectorSize), 4);
}

// Physical UE4M3 element counts for Sm1xxBlockScaledConfig<16>.
inline constexpr std::size_t scale_a_elements(int m, int /*n*/, int k) {
  return static_cast<std::size_t>(align_up(m, kScaleMNAlignment)) *
         static_cast<std::size_t>(scale_k_padded(k));
}

inline constexpr std::size_t scale_b_elements(int /*m*/, int n, int k) {
  return static_cast<std::size_t>(align_up(n, kScaleMNAlignment)) *
         static_cast<std::size_t>(scale_k_padded(k));
}

enum class GemmStatus : int {
  kSuccess = 0,
  kInvalidArgument,
  kUnsupportedDevice,
  kInsufficientWorkspace,
  kCutlassNotSupported,
  kCutlassError,
  kCudaError,
};

const char* gemm_status_string(GemmStatus status);

// Custom CuTe implementation. This launches a repository-owned __global__ kernel,
// uses a persistent CTA schedule, and executes the SM120 block-scaled MMA via
// cute::gemm. It does not use CUTLASS GemmUniversal or its device adapter.
GemmStatus nvfp4_cute_gemm_sm120(
    int m, int n, int k,
    const void* a, const void* b,
    const void* sfa, const void* sfb,
    half* c, cudaStream_t stream = nullptr);

// CUTLASS Collective reference retained for correctness and performance
// comparisons against both the custom CuTe kernel and cuBLASLt.
std::size_t nvfp4_cutlass_gemm_workspace_size_sm120(int m, int n, int k);

GemmStatus nvfp4_cutlass_gemm_sm120(
    int m, int n, int k,
    const void* a, const void* b,
    const void* sfa, const void* sfb,
    half* c,
    void* workspace, std::size_t workspace_bytes,
    cudaStream_t stream = nullptr);

// Workspace size for the default custom CuTe implementation. A return value of
// zero is valid. Invalid dimensions also return zero and are rejected by the
// launch function.
std::size_t nvfp4_gemm_workspace_size_sm120(int m, int n, int k);

// Default custom CuTe NVFP4 GEMM on SM120:
//
//   C[M,N] = (A_payload * SFA) @ (B_payload * SFB)^T
//
// A is packed row-major [M,K]. B is packed row-major [N,K]; this is the TN
// operand layout required by SM120 block-scaled Tensor Core instructions.
// SFA/SFB are raw UE4M3 bytes in CUTLASS Sm1xxBlockScaledConfig<16> physical
// layout. C is row-major FP16 [M,N]. All pointers are device pointers.
GemmStatus nvfp4_gemm_sm120(
    int m, int n, int k,
    const void* a, const void* b,
    const void* sfa, const void* sfb,
    half* c,
    void* workspace, std::size_t workspace_bytes,
    cudaStream_t stream = nullptr);

template <int N, int K>
inline GemmStatus nvfp4_gemm_sm120(
    int m,
    const void* a, const void* b,
    const void* sfa, const void* sfb,
    half* c,
    void* workspace, std::size_t workspace_bytes,
    cudaStream_t stream = nullptr) {
  return nvfp4_gemm_sm120(
      m, N, K, a, b, sfa, sfb, c, workspace, workspace_bytes, stream);
}

}  // namespace sm120_nvfp4
