#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include "cute/tensor.hpp"
#include "cute/atom/mma_traits_sm120.hpp"
#include "cutlass/float_subbyte.h"
#include "cutlass/version.h"

#if CUTLASS_VERSION < 420
#error "Sparse MLA requires CUTLASS 4.2+ SM120 block-scaled CuTe atoms"
#endif

namespace sm120_nvfp4::sparse_mla {

using NvOp = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
    cutlass::float_e2m1_t, cutlass::float_e2m1_t, float,
    cutlass::float_ue4m3_t, 16>;
using BfOp = cute::SM80_16x8x16_F32BF16BF16F32_TN;
using NvTraits = cute::MMA_Traits<NvOp>;
using BfTraits = cute::MMA_Traits<BfOp>;

// Both instructions use the same (thread,value)->(row,column) accumulator map.
CUTE_DEVICE int result_row(int lane, int value) {
  return NvTraits::CLayout{}(lane, value) % 16;
}
CUTE_DEVICE int result_column(int lane, int value) {
  return NvTraits::CLayout{}(lane, value) / 16;
}

// One warp's 16x8x64 block-scaled product. CuTe owns the register layouts and
// instruction dispatch; row-major shared operands need no global repacking.
// A and B are stored (M,K) and (N,K), packed along K. The logical SF fragments
// broadcast each of four physical scale bytes to sixteen K coordinates.
template <int AStride, int BStride, int ASStride, int BSStride>
CUTE_DEVICE void nv_mma(const std::uint8_t* ap, const std::uint8_t* bp,
                        const std::uint8_t* asp, const std::uint8_t* bsp,
                        float (&acc)[4], int lane) {
  using namespace cute;
  static_assert(AStride % 4 == 0 && BStride % 4 == 0,
                "Packed FP4 fragment loads require 4-byte row alignment");
  auto a = make_tensor<uint4_t>(Layout<_32>{});
  auto b = make_tensor<uint4_t>(Layout<_16>{});
  auto a32 = recast<std::uint32_t>(a);
  auto b32 = recast<std::uint32_t>(b);
  using SfLayout = Layout<Shape<Shape<_16, _4>>, Stride<Stride<_0, _1>>>;
  auto sa = make_tensor<cutlass::float_ue4m3_t>(SfLayout{});
  auto sb = make_tensor<cutlass::float_ue4m3_t>(SfLayout{});
  auto c = make_tensor(make_rmem_ptr(acc), Layout<_4>{});
  // Eight consecutive fragment nibbles are one aligned word along K in
  // these SM120 layouts. All callers provide 4-byte-aligned shared bases.
  // Load the packed register bits directly instead of extracting/repacking
  // individual nibbles. The existing row layout (and bank mapping) is kept.
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int coord = NvTraits::ALayout{}(lane, i * 8);
    a32(i) = *reinterpret_cast<const std::uint32_t*>(
        ap + (coord % 16) * AStride + (coord / 16) / 2);
  }
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int coord = NvTraits::BLayout{}(lane, i * 8);
    b32(i) = *reinterpret_cast<const std::uint32_t*>(
        bp + (coord % 8) * BStride + (coord / 8) / 2);
  }
  int ar = NvTraits::SFALayout{}(lane, 0) % 16;
  int br = NvTraits::SFBLayout{}(lane, 0) % 8;
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    sa(g * 16).raw() = asp[ar * ASStride + g];
    sb(g * 16).raw() = bsp[br * BSStride + g];
  }
  cute::gemm(MMA_Atom<NvOp>{}, make_zip_tensor(a, sa), make_zip_tensor(b, sb), c);
}

// B can be token-major (QK) or its logical transpose (PV). Element strides
// describe that choice without materializing a transposed BF16 cache.
template <int ARowStride, int BRowStride, int BKStride, bool PackedA = false>
CUTE_DEVICE void bf_mma(const __nv_bfloat16* ap, const __nv_bfloat16* bp,
                        float (&acc)[4], int lane) {
  using namespace cute;
  auto a = make_tensor<cutlass::bfloat16_t>(Layout<_8>{});
  auto b = make_tensor<cutlass::bfloat16_t>(Layout<_4>{});
  auto c = make_tensor(make_rmem_ptr(acc), Layout<_4>{});
  if constexpr (PackedA) {
    static_assert(ARowStride % 2 == 0, "Packed BF16 A loads need even row stride");
    // Q and padded shared-weight callers provide 16-byte-aligned bases.
    // ALayout maps each consecutive fragment pair to adjacent BF16 values
    // at an even K, satisfying the 4-byte alignment of the packed load.
    auto a32 = recast<std::uint32_t>(a);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int coord = BfTraits::ALayout{}(lane, i * 2);
      a32(i) = *reinterpret_cast<const std::uint32_t*>(
          ap + (coord % 16) * ARowStride + coord / 16);
    }
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int coord = BfTraits::ALayout{}(lane, i);
      a(i) = cutlass::bfloat16_t(__bfloat162float(ap[(coord % 16) * ARowStride + coord / 16]));
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int coord = BfTraits::BLayout{}(lane, i);
    b(i) = cutlass::bfloat16_t(__bfloat162float(bp[(coord % 8) * BRowStride + (coord / 8) * BKStride]));
  }
  cute::gemm(MMA_Atom<BfOp>{}, a, b, c);
}

}  // namespace sm120_nvfp4::sparse_mla
