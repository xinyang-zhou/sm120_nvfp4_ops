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
  auto a = make_tensor<uint4_t>(Layout<_32>{});
  auto b = make_tensor<uint4_t>(Layout<_16>{});
  clear(a);
  clear(b);
  using SfLayout = Layout<Shape<Shape<_16, _4>>, Stride<Stride<_0, _1>>>;
  auto sa = make_tensor<cutlass::float_ue4m3_t>(SfLayout{});
  auto sb = make_tensor<cutlass::float_ue4m3_t>(SfLayout{});
  auto c = make_tensor(make_rmem_ptr(acc), Layout<_4>{});
  auto am = make_tensor(make_smem_ptr(ap),
                        make_layout(make_shape(_16{}, _32{}),
                                    make_stride(Int<AStride>{}, _1{})));
  auto bm = make_tensor(make_smem_ptr(bp),
                        make_layout(make_shape(_8{}, _32{}),
                                    make_stride(Int<BStride>{}, _1{})));
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    int coord = NvTraits::ALayout{}(lane, i);
    int k = coord / 16;
    a(i) = uint4_t((am(coord % 16, k / 2) >> (4 * (k & 1))) & 15);
  }
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    int coord = NvTraits::BLayout{}(lane, i);
    int k = coord / 8;
    b(i) = uint4_t((bm(coord % 8, k / 2) >> (4 * (k & 1))) & 15);
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
template <int ARowStride, int BRowStride, int BKStride>
CUTE_DEVICE void bf_mma(const __nv_bfloat16* ap, const __nv_bfloat16* bp,
                        float (&acc)[4], int lane) {
  using namespace cute;
  auto a = make_tensor<cutlass::bfloat16_t>(Layout<_8>{});
  auto b = make_tensor<cutlass::bfloat16_t>(Layout<_4>{});
  auto c = make_tensor(make_rmem_ptr(acc), Layout<_4>{});
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int coord = BfTraits::ALayout{}(lane, i);
    a(i) = cutlass::bfloat16_t(__bfloat162float(ap[(coord % 16) * ARowStride + coord / 16]));
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int coord = BfTraits::BLayout{}(lane, i);
    b(i) = cutlass::bfloat16_t(__bfloat162float(bp[(coord % 8) * BRowStride + (coord / 8) * BKStride]));
  }
  cute::gemm(MMA_Atom<BfOp>{}, a, b, c);
}

}  // namespace sm120_nvfp4::sparse_mla
