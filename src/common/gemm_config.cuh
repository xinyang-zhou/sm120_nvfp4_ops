#ifndef SM120_NVFP4_GEMM_CONFIG_CUH_
#define SM120_NVFP4_GEMM_CONFIG_CUH_

#include <cstdint>
#include <tuple>
#include <type_traits>

#include "cute/tensor.hpp"
#include "cutlass/float_subbyte.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp"
#include "cutlass/version.h"

#if CUTLASS_VERSION < 420
#error "SM120 NVFP4 GEMM requires CUTLASS 4.2 or newer (SM120 block-scaled MMA support)"
#endif

namespace sm120_nvfp4 {
namespace detail {

using namespace cute;  // NOLINT

// SM120 NVFP4 uses a fixed 128x4 physical scale-factor block.  The CUTLASS
// collective below is used as the source of truth for the shared-memory scale
// layouts and copy atoms.  TiledMma is written explicitly to make the selected
// rr_blockscaled SM120 instruction visible in this repository's config layer.
template <typename Tout_, int kTileM_, int kTileN_, int kTileK_, int kStage_>
struct Nvfp4GemmConfig {
  using Tin = cutlass::float_e2m1_t;
  using Tout = Tout_;
  using TS = cutlass::float_ue4m3_t;

  static constexpr int kSFVectorSize = 16;
  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;
  static constexpr int kStage = kStage_;

  static_assert(kTileM % 128 == 0,
                "SM120 NVFP4 SFA shared layout requires TileM to be a multiple of 128");
  static_assert(kTileN % 128 == 0,
                "SM120 NVFP4 SFB shared layout requires TileN to be a multiple of 128");
  static_assert(kTileK % 64 == 0,
                "SM120 NVFP4 MMA and its 4-scale block require TileK to be a multiple of 64");
  static_assert(kStage >= 2, "warp-specialized NVFP4 GEMM needs at least two stages");

  using TileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>;
  using ClusterShape = Shape<_1, _1, _1>;

  using MmaAtom = decltype(cute::rr_blockscaled_op_selector_sm120<
                           Tin, Tin, float, TS, kSFVectorSize, false>());
  using AtomLayoutMNK = Layout<Shape<_4, _2, _1>>;
  using PermTileM = Int<(kTileM < 128 ? kTileM : 128)>;
  using PermTileN = Layout<Shape<_8, _2, _2>, Stride<_1, _16, _8>>;
  using PermTileK = _64;
  using TiledMma = decltype(cute::make_tiled_mma(
      MmaAtom{}, AtomLayoutMNK{}, Tile<PermTileM, PermTileN, PermTileK>{}));

  using ElementA = cutlass::nv_float4_t<Tin>;
  using ElementB = cutlass::nv_float4_t<Tin>;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
      ElementA, cutlass::layout::RowMajor, 32,
      ElementB, cutlass::layout::ColumnMajor, 32,
      float, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCount<kStage>,
      cutlass::gemm::KernelTmaWarpSpecializedNvf4Sm120>::CollectiveOp;

  static_assert(std::is_same_v<TiledMma, typename CollectiveMainloop::TiledMma>,
                "explicit SM120 NVFP4 TiledMma must match CUTLASS CollectiveBuilder");

  using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<kSFVectorSize>;
  using LayoutSFA = decltype(Sm1xxBlkScaledConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(Sm1xxBlkScaledConfig::deduce_layoutSFB());

  using SLayoutX = typename CollectiveMainloop::SmemLayoutA;
  using SLayoutW = typename CollectiveMainloop::SmemLayoutB;
  using SLayoutSFA = typename CollectiveMainloop::SmemLayoutSFA;
  using SLayoutSFB = typename CollectiveMainloop::SmemLayoutSFB;
  using SmemCopyAtomX = typename CollectiveMainloop::SmemCopyAtomA;
  using SmemCopyAtomW = typename CollectiveMainloop::SmemCopyAtomB;
  using SmemCopyAtomSFA = typename CollectiveMainloop::SmemCopyAtomSFA;
  using SmemCopyAtomSFB = typename CollectiveMainloop::SmemCopyAtomSFB;

  // Keep the TMA types independent of runtime problem sizes. CUTLASS defines
  // these from placeholder tensors whose dynamic modes are int32_t. Letting
  // make_tma_copy() deduce the types from a runtime tensor can change the
  // kernel specialization (for example, static-vs-dynamic L modes), leaving
  // the host launch stub without a matching device function.
  using TmaX = typename CollectiveMainloop::Params::TMA_A;
  using TmaW = typename CollectiveMainloop::Params::TMA_B;
  using TmaSFA = typename CollectiveMainloop::Params::TMA_SFA;
  using TmaSFB = typename CollectiveMainloop::Params::TMA_SFB;
  using StrideX = typename CollectiveMainloop::StrideA;
  using StrideW = typename CollectiveMainloop::StrideB;
  using TmaInternalElementX = typename CollectiveMainloop::TmaInternalElementA;
  using TmaInternalElementW = typename CollectiveMainloop::TmaInternalElementB;

  struct TmaBundle {
    TmaX x;
    TmaW w;
    TmaSFA sfa;
    TmaSFB sfb;
  };

  struct alignas(1024) TensorStorage {
    alignas(1024) ArrayEngine<Tin, cosize_v<SLayoutX>> smem_X;
    alignas(1024) ArrayEngine<Tin, cosize_v<SLayoutW>> smem_W;
    alignas(128) ArrayEngine<TS, cosize_v<SLayoutSFA>> smem_SFA;
    alignas(128) ArrayEngine<TS, cosize_v<SLayoutSFB>> smem_SFB;
  };

  static constexpr uint32_t kTmaTransactionBytes =
      CollectiveMainloop::TmaTransactionBytes;

  CUTE_HOST_DEVICE static auto make_layout_sfa(int m, int n, int k, int groups = 1) {
    return Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        make_shape(m, n, k, groups));
  }

  CUTE_HOST_DEVICE static auto make_layout_sfb(int m, int n, int k, int groups = 1) {
    return Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        make_shape(m, n, k, groups));
  }

  template <typename TX, typename TW, typename TXS, typename TWS>
  TmaBundle get_tma(TX x, TW w, TXS xs, TWS ws) const {
    TmaX tma_x = make_tma_copy(
        SM90_TMA_LOAD{}, x, SLayoutX{}(_, _, _0{}),
        make_shape(Int<kTileM>{}, Int<kTileK>{}), _1{});
    TmaW tma_w = make_tma_copy(
        SM90_TMA_LOAD{}, w, SLayoutW{}(_, _, _0{}),
        make_shape(Int<kTileN>{}, Int<kTileK>{}), _1{});
    TmaSFA tma_xs = make_tma_copy<uint16_t>(
        SM90_TMA_LOAD{}, xs, SLayoutSFA{}(_, _, _0{}),
        make_shape(Int<kTileM>{}, Int<kTileK>{}), _1{});
    TmaSFB tma_ws = make_tma_copy<uint16_t>(
        SM90_TMA_LOAD{}, ws, SLayoutSFB{}(_, _, _0{}),
        make_shape(Int<kTileN>{}, Int<kTileK>{}), _1{});
    return {tma_x, tma_w, tma_xs, tma_ws};
  }

  static constexpr int get_shm_size() { return sizeof(TensorStorage); }
};

}  // namespace detail
}  // namespace sm120_nvfp4

#endif  // SM120_NVFP4_GEMM_CONFIG_CUH_
