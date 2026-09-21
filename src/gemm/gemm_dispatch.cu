#include "sm120_nvfp4/gemm.hpp"

#include <cstddef>

#include "gemm/specialized/m128_gemm.hpp"
#include "gemm/specialized/m256_gemm.hpp"

namespace sm120_nvfp4 {
namespace {

constexpr int kSingleGroup = 1;
constexpr int kM128SplitK = 4;
constexpr int kM256SplitK = 2;

std::size_t specialized_workspace_size(
    Nvfp4GemmPath path, int m, int n, int k) {
  switch (path) {
    case Nvfp4GemmPath::kM128SplitK4:
      return gemm_specialized::m128::workspace_size(
          kSingleGroup, m, n, k, kM128SplitK);
    case Nvfp4GemmPath::kM256SplitK2:
      return gemm_specialized::m256::workspace_size(
          kSingleGroup, m, n, k, kM256SplitK);
    case Nvfp4GemmPath::kGenericCute:
      return 0;
  }
  return 0;
}

}  // namespace

const char* nvfp4_gemm_path_string(Nvfp4GemmPath path) {
  switch (path) {
    case Nvfp4GemmPath::kGenericCute:
      return "generic_cute";
    case Nvfp4GemmPath::kM128SplitK4:
      return "m128_splitk4";
    case Nvfp4GemmPath::kM256SplitK2:
      return "m256_splitk2";
  }
  return "unknown";
}

Nvfp4GemmPath nvfp4_gemm_path_sm120(int m, int n, int k) {
  if (m == 128 &&
      gemm_specialized::m128::workspace_size(
          kSingleGroup, m, n, k, kM128SplitK) != 0) {
    return Nvfp4GemmPath::kM128SplitK4;
  }
  if (m == 256 &&
      gemm_specialized::m256::workspace_size(
          kSingleGroup, m, n, k, kM256SplitK) != 0) {
    return Nvfp4GemmPath::kM256SplitK2;
  }
  return Nvfp4GemmPath::kGenericCute;
}

std::size_t nvfp4_gemm_workspace_size_sm120(int m, int n, int k) {
  const Nvfp4GemmPath path = nvfp4_gemm_path_sm120(m, n, k);
  return specialized_workspace_size(path, m, n, k);
}

GemmStatus nvfp4_gemm_sm120(
    int m, int n, int k, const void* a, const void* b,
    const void* sfa, const void* sfb, half* c,
    void* workspace, std::size_t workspace_bytes,
    cudaStream_t stream) {
  const Nvfp4GemmPath path = nvfp4_gemm_path_sm120(m, n, k);
  const std::size_t required_workspace =
      specialized_workspace_size(path, m, n, k);
  if (required_workspace != 0 &&
      (workspace == nullptr || workspace_bytes < required_workspace)) {
    return GemmStatus::kInsufficientWorkspace;
  }

  switch (path) {
    case Nvfp4GemmPath::kM128SplitK4:
      return gemm_specialized::m128::launch(
          kSingleGroup, m, n, k, kM128SplitK,
          a, b, sfa, sfb, c, workspace, workspace_bytes, stream);
    case Nvfp4GemmPath::kM256SplitK2:
      return gemm_specialized::m256::launch(
          kSingleGroup, m, n, k, kM256SplitK,
          a, b, sfa, sfb, c, workspace, workspace_bytes, stream);
    case Nvfp4GemmPath::kGenericCute:
      return nvfp4_cute_gemm_sm120(
          m, n, k, a, b, sfa, sfb, c, stream);
  }
  return GemmStatus::kInvalidArgument;
}

}  // namespace sm120_nvfp4
