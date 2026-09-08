#include <cstdint>
#include <stdexcept>
#include <string>

#include "cute/tensor.hpp"
#include "common/gemm_config.cuh"
#include "sm120_nvfp4/grouped_gemm.hpp"
#include "grouped_gemm/kernels.cuh"
#include "common/device_info.hpp"

namespace sm120_nvfp4 {
namespace grouped_gemm {

namespace {

void check_cuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

}  // namespace

template <int kTileM, int kTileN, int kTileK, int kStage>
void launch_group_gemm_nvfp4(
    void *y_ptr, const void *x_ptr, const void *w_ptr,
    const void *seqlens_ptr, const void *cu_seqlens_ptr,
    const void *x_scale_ptr, const void *w_scale_ptr,
    void *tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr,
    int num_group, int m, int n, int k, int m_scale_pad,
    bool update_tma, cudaStream_t stream) {
  using namespace cute;  // NOLINT
  using Tout = cute::half_t;
  using Config = sm120_nvfp4::detail::Nvfp4GemmConfig<Tout, kTileM, kTileN, kTileK, kStage>;
  using TmaInternalElementX = typename Config::TmaInternalElementX;
  using TmaInternalElementW = typename Config::TmaInternalElementW;
  using TS = typename Config::TS;

  // Kept in the ABI for existing callers. Per-group TensorMaps contain
  // invocation-specific base addresses and shapes, so reusing them across
  // calls is not safe.
  (void)update_tma;

  Config config;

  auto X = make_tensor(
      recast_ptr<TmaInternalElementX>(x_ptr),
      make_shape(int32_t(m), int32_t(k), int32_t(1)),
      typename Config::StrideX{int64_t(k), Int<1>{}, int64_t(m) * k});
  auto W = make_tensor(
      recast_ptr<TmaInternalElementW>(w_ptr),
      make_shape(int32_t(n), int32_t(k), int32_t(num_group)),
      typename Config::StrideW{int64_t(k), Int<1>{}, int64_t(n) * k});

  auto layout_sfa = Config::make_layout_sfa(m_scale_pad, n, k, 1);
  auto layout_sfb = Config::make_layout_sfb(m_scale_pad, n, k, num_group);
  auto SFA = make_tensor(
      reinterpret_cast<const TS *>(x_scale_ptr), layout_sfa);
  auto SFB = make_tensor(
      reinterpret_cast<const TS *>(w_scale_ptr), layout_sfb);

  auto [tma_x, tma_w, tma_sfa, tma_sfb] =
      config.get_tma(X, W, SFA, SFB);

  auto *tma_xysfa = static_cast<cute::TmaDescriptor *>(tmas_ptr);
  auto *tiles = static_cast<int *>(tiles_ptr);
  auto *cu_tiles = static_cast<int *>(cu_tiles_ptr);

  constexpr int kThreadPerBlock = 32;
  constexpr int kGroupPerThread = 8;
  int64_t scale_k =
      (static_cast<int64_t>(k) + Config::kSFVectorSize - 1) /
      Config::kSFVectorSize;
  int64_t scale_k_padded = (scale_k + 3) / 4 * 4;
  int64_t sfa_group_elements = static_cast<int64_t>(m_scale_pad) * scale_k_padded;

  kernels::update_grouped_tma_x_nvfp4<Config>
      <<<num_group, 32, 0, stream>>>(
          *tma_x.get_tma_descriptor(), tma_xysfa,
          static_cast<const uint8_t *>(x_ptr),
          static_cast<const int *>(seqlens_ptr),
          static_cast<const int *>(cu_seqlens_ptr), k);
  check_cuda(cudaPeekAtLastError(), "update_grouped_tma_x_nvfp4 launch");

  kernels::update_grouped_tma_sfa_nvfp4
      <<<num_group, 32, 0, stream>>>(
          *tma_sfa.get_tma_descriptor(), tma_xysfa,
          static_cast<const uint8_t *>(x_scale_ptr), sfa_group_elements);
  check_cuda(cudaPeekAtLastError(), "update_grouped_tma_sfa_nvfp4 launch");

  kernels::initialize_static_tma_nvfp4<<<1, 1, 0, stream>>>(
      *tma_w.get_tma_descriptor(), *tma_sfb.get_tma_descriptor(),
      tma_xysfa + num_group * 3);
  check_cuda(cudaPeekAtLastError(), "initialize_static_tma_nvfp4 launch");

  // TensorMaps and tile metadata are all per-call scratch state.
  kernels::update_grouped_tiles_nvfp4<
      kTileM, kGroupPerThread, kThreadPerBlock>
      <<<1, kThreadPerBlock, 0, stream>>>(
          static_cast<const int *>(seqlens_ptr), tiles, cu_tiles, num_group);
  check_cuda(cudaPeekAtLastError(), "update_grouped_tiles_nvfp4 launch");

  int num_sm = get_sm_count();
  int num_tile_n = n / kTileN + (n % kTileN != 0);
  cutlass::FastDivmod flat_divider(num_tile_n);

  int shm_size = config.get_shm_size() + sizeof(int) * num_group;
  dim3 block(384);
  dim3 grid(num_sm);

  auto kernel = kernels::group_gemm_nvfp4_kernel<Config>;
  check_cuda(cudaFuncSetAttribute(
                 kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size),
             "cudaFuncSetAttribute(group_gemm_nvfp4_kernel)");
  kernel<<<grid, block, shm_size, stream>>>(
      tma_xysfa, static_cast<Tout *>(y_ptr),
      static_cast<const int *>(seqlens_ptr),
      static_cast<const int *>(cu_seqlens_ptr), tiles, cu_tiles,
      num_group, m_scale_pad, n, k, flat_divider);
  check_cuda(cudaPeekAtLastError(), "group_gemm_nvfp4_kernel launch");
}

void group_gemm_nvfp4_async(
    void *y_ptr, const void *x_ptr, const void *w_ptr,
    const void *seqlens_ptr, const void *cu_seqlens_ptr,
    const void *x_scale_ptr, const void *w_scale_ptr,
    void *tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr,
    int num_group, int m, int n, int k, int m_scale_pad,
    int num_seq_per_group_avg, bool update_tma, cudaStream_t stream) {
  // The first implementation intentionally keeps one CUTLASS-valid scale tile
  // configuration.  num_seq_per_group_avg stays in the public ABI so legal
  // SM120 ping-pong/cooperative variants can be added without changing callers.
  (void)num_seq_per_group_avg;
  constexpr int kTileM = 128;
  constexpr int kTileN = 128;
  constexpr int kTileK = 128;
  constexpr int kStage = 3;

  launch_group_gemm_nvfp4<kTileM, kTileN, kTileK, kStage>(
      y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr,
      x_scale_ptr, w_scale_ptr, tmas_ptr, tiles_ptr, cu_tiles_ptr,
      num_group, m, n, k, m_scale_pad, update_tma, stream);
}

}  // namespace grouped_gemm
}  // namespace sm120_nvfp4
