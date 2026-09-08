#ifndef SM120_NVFP4_GROUPED_GEMM_KERNELS_CUH_
#define SM120_NVFP4_GROUPED_GEMM_KERNELS_CUH_

#include <cstdint>
#include <type_traits>

#include <cub/cub.cuh>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp"

namespace sm120_nvfp4 {
namespace grouped_gemm {
namespace kernels {

__device__ __forceinline__ void publish_tma_descriptor(
    cute::TmaDescriptor *gmem_desc, const cute::TmaDescriptor &smem_desc) {
  auto *dst = reinterpret_cast<uint64_t *>(gmem_desc);
  auto *src = reinterpret_cast<const uint64_t *>(&smem_desc);
#pragma unroll
  for (int i = 0; i < CU_TENSOR_MAP_NUM_QWORDS; ++i) {
    dst[i] = src[i];
  }
  cute::tma_descriptor_fence_release();
}

template <typename Tma, typename GTensor>
__device__ __forceinline__ void update_subbyte_tma_gtensor(
    cute::TmaDescriptor &smem_tma_desc, const GTensor &gtensor) {
  cute::array<uint32_t, 5> shape{1, 1, 1, 1, 1};
  cute::array<uint64_t, 5> stride{0, 0, 0, 0, 0};
  cute::detail::fill_tma_gmem_shape_stride(Tma{}, gtensor, shape, stride);

  const void *gmem_ptr = cute::raw_pointer_cast(gtensor.data());
  cute::tma_descriptor_replace_addr_in_shared_mem(smem_tma_desc, gmem_ptr);
  using T = typename GTensor::value_type;
  for (auto &s : stride) {
    s = (s * cute::sizeof_bits_v<T>) / 8;
  }
  cute::tma_descriptor_replace_dims_strides_in_shared_mem(
      smem_tma_desc, shape, stride);
}

__device__ __forceinline__ void get_next_nvfp4_tile_horizon(
    const int *tiles_ptr, int iblock, int num_group, int &igroup,
    int &itile_m, int &itile_n, int &sum_tile_m,
    cutlass::FastDivmod flat_divider) {
  int itile_m_total = 0;
  flat_divider(itile_m_total, itile_n, iblock);

  for (int i = igroup; i < num_group; ++i) {
    int num_tile_m = tiles_ptr[i];
    sum_tile_m += num_tile_m;
    if (itile_m_total < sum_tile_m) {
      igroup = i;
      sum_tile_m -= num_tile_m;
      itile_m = itile_m_total - sum_tile_m;
      return;
    }
  }
  igroup = -1;
}

template <int kTileM, int kGroupPerThread, int kThreadPerBlock>
__global__ void update_grouped_tiles_nvfp4(
    const int *seqlens_ptr, int *tiles_ptr, int *cu_tiles_ptr,
    int num_group) {
  int idx = threadIdx.x;
  int tiles[kGroupPerThread];

#pragma unroll
  for (int i = 0; i < kGroupPerThread; ++i) {
    int group = idx * kGroupPerThread + i;
    if (group < num_group) {
      int group_m = seqlens_ptr[group];
      tiles[i] = group_m / kTileM + (group_m % kTileM != 0);
      tiles_ptr[group] = tiles[i];
    } else {
      tiles[i] = 0;
    }
  }

  using BlockScan = cub::BlockScan<int, kThreadPerBlock>;
  __shared__ typename BlockScan::TempStorage temp_storage;
  int block_aggregate = 0;
  BlockScan(temp_storage).ExclusiveSum(tiles, tiles, block_aggregate);

#pragma unroll
  for (int i = 0; i < kGroupPerThread; ++i) {
    int group = idx * kGroupPerThread + i;
    if (group < num_group) {
      cu_tiles_ptr[group] = tiles[i];
    }
  }
  if (idx == 0) {
    cu_tiles_ptr[num_group] = block_aggregate;
  }
}

template <typename Config>
__global__ void update_grouped_tma_x_nvfp4(
    const __grid_constant__ cute::TmaDescriptor tma_template,
    cute::TmaDescriptor *tma_xysfa, const uint8_t *x_ptr,
    const int *seqlens_ptr, const int *cu_seqlens_ptr, int k) {
  using namespace cute;  // NOLINT
  using TmaX = typename Config::TmaX;
  using TmaInternalElementX = typename Config::TmaInternalElementX;
  if (threadIdx.x != 0) {
    return;
  }

  int group = blockIdx.x;
  int descriptor_m = seqlens_ptr[group] > 0 ? seqlens_ptr[group] : 1;
  int64_t cu_seqlen = cu_seqlens_ptr[group];
  const uint8_t *group_ptr = x_ptr + cu_seqlen * (k / 2);

  __shared__ cute::TmaDescriptor smem_tma_desc;
  smem_tma_desc = tma_template;
  auto gX = make_tensor(
      recast_ptr<TmaInternalElementX>(group_ptr),
      make_shape(int32_t(descriptor_m), int32_t(k), int32_t(1)),
      typename Config::StrideX{int64_t(k), Int<1>{},
                               int64_t(descriptor_m) * k});
  update_subbyte_tma_gtensor<TmaX>(smem_tma_desc, gX);
  cute::tma_desc_commit_group();
  cute::tma_desc_wait_group();
  publish_tma_descriptor(tma_xysfa + group * 3, smem_tma_desc);
}

__global__ void update_grouped_tma_sfa_nvfp4(
    const __grid_constant__ cute::TmaDescriptor tma_template,
    cute::TmaDescriptor *tma_xysfa, const uint8_t *x_scale_ptr,
    int64_t sfa_group_elements) {
  if (threadIdx.x != 0) {
    return;
  }

  int group = blockIdx.x;
  const uint8_t *group_ptr = x_scale_ptr + group * sfa_group_elements;

  __shared__ cute::TmaDescriptor smem_tma_desc;
  smem_tma_desc = tma_template;
  cute::tma_descriptor_replace_addr_in_shared_mem(smem_tma_desc, group_ptr);
  cute::tma_desc_commit_group();
  cute::tma_desc_wait_group();
  publish_tma_descriptor(tma_xysfa + group * 3 + 2, smem_tma_desc);
}

__global__ void initialize_static_tma_nvfp4(
    const __grid_constant__ cute::TmaDescriptor tma_w,
    const __grid_constant__ cute::TmaDescriptor tma_sfb,
    cute::TmaDescriptor *static_tmas) {
  __shared__ cute::TmaDescriptor smem_tmas[2];
  if (threadIdx.x == 0) {
    smem_tmas[0] = tma_w;
    smem_tmas[1] = tma_sfb;
    cute::tma_desc_commit_group();
    cute::tma_desc_wait_group();
    publish_tma_descriptor(static_tmas, smem_tmas[0]);
    publish_tma_descriptor(static_tmas + 1, smem_tmas[1]);
  }
}

template <typename Config>
__global__ void __launch_bounds__(384, 1) group_gemm_nvfp4_kernel(
    cute::TmaDescriptor *tma_xysfa,
    typename Config::Tout *y_ptr,
    const int *seqlens_ptr, const int *cu_seqlens_ptr,
    const int *tiles_ptr, const int *cu_tiles_ptr,
    int num_group, int m_scale_pad, int n, int k,
    cutlass::FastDivmod flat_divider) {
  using namespace cute;  // NOLINT
  using Tin = typename Config::Tin;
  using Tout = typename Config::Tout;
  using TS = typename Config::TS;
  using TmaX = typename Config::TmaX;
  using TmaW = typename Config::TmaW;
  using TmaSFA = typename Config::TmaSFA;
  using TmaSFB = typename Config::TmaSFB;
  using TiledMma = typename Config::TiledMma;
  using CollectiveMainloop = typename Config::CollectiveMainloop;
  using TensorStorage = typename Config::TensorStorage;
  using SLayoutX = typename Config::SLayoutX;
  using SLayoutW = typename Config::SLayoutW;
  using SLayoutSFA = typename Config::SLayoutSFA;
  using SLayoutSFB = typename Config::SLayoutSFB;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kStage = Config::kStage;
  constexpr int kMathThreads = size(TiledMma{});
  static_assert(kMathThreads == 256,
                "cooperative SM120 NVFP4 config must use 256 math threads");

  int thread_idx = threadIdx.x;
  int elected = cute::elect_one_sync();

  __shared__ uint64_t writable[kStage];
  __shared__ uint64_t readable[kStage];

  extern __shared__ uint8_t smem[] alignas(1024);
  TensorStorage &storage = *reinterpret_cast<TensorStorage *>(smem);
  int *shm_tiles = reinterpret_cast<int *>(smem + sizeof(TensorStorage));

  auto sX = make_tensor(make_smem_ptr(storage.smem_X.begin()), SLayoutX{});
  auto sW = make_tensor(make_smem_ptr(storage.smem_W.begin()), SLayoutW{});
  auto sSFA = make_tensor(make_smem_ptr(storage.smem_SFA.begin()), SLayoutSFA{});
  auto sSFB = make_tensor(make_smem_ptr(storage.smem_SFB.begin()), SLayoutSFB{});

  TmaX tma_x;
  TmaW tma_w;
  TmaSFA tma_sfa;
  TmaSFB tma_sfb;

  auto layout_sfa = Config::make_layout_sfa(m_scale_pad, n, k, 1);
  auto layout_sfb = Config::make_layout_sfb(m_scale_pad, n, k, num_group);

  auto mX = tma_x.get_tma_tensor(make_shape(m_scale_pad, k, Int<1>{}));
  auto mW = tma_w.get_tma_tensor(make_shape(n, k, num_group));
  auto mSFA = tma_sfa.get_tma_tensor(shape(layout_sfa));
  auto mSFB = tma_sfb.get_tma_tensor(shape(layout_sfb));

  using X = Underscore;
  using TileShape = typename Config::TileShape;
  auto gX_mkl = local_tile(mX, TileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});
  auto gW_nkl = local_tile(mW, TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});
  auto gSFA_mkl = local_tile(mSFA, TileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});
  auto gSFB_nkl = local_tile(mSFB, TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});

  auto btma_x = tma_x.get_slice(0);
  auto btma_w = tma_w.get_slice(0);
  auto btma_sfa = tma_sfa.get_slice(0);
  auto btma_sfb = tma_sfb.get_slice(0);
  auto tXsX = btma_x.partition_D(sX);
  auto tWsW = btma_w.partition_D(sW);
  auto tSFAsSFA = btma_sfa.partition_D(sSFA);
  auto tSFBsSFB = btma_sfb.partition_D(sSFB);

  if (thread_idx < kStage) {
    initialize_barrier(readable[thread_idx], 1);
    initialize_barrier(writable[thread_idx], kMathThreads);
  }

  for (int i = thread_idx; i < num_group; i += blockDim.x) {
    shm_tiles[i] = tiles_ptr[i];
  }
  __syncthreads();

  if (thread_idx >= kMathThreads) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    int load_thread = thread_idx - kMathThreads;
    int load_warp = __shfl_sync(0xffffffff, load_thread / 32, 0);
    bool is_load_leader = load_warp == 0 && elected;

    if (is_load_leader) {
      auto *td_w = tma_xysfa + num_group * 3;
      auto *td_sfb = td_w + 1;
      tma_descriptor_fence_acquire(td_w);
      tma_descriptor_fence_acquire(td_sfb);

      int write_stage = 0;
      int write_phase = 1;
      int iblock = blockIdx.x;
      int group = 0;
      int sum_tile_m = 0;
      int itile_m = 0;
      int itile_n = 0;
      int num_tile_k = k / kTileK + (k % kTileK != 0);

      while (true) {
        get_next_nvfp4_tile_horizon(
            shm_tiles, iblock, num_group, group,
            itile_m, itile_n, sum_tile_m, flat_divider);
        if (group < 0) {
          break;
        }
        iblock += gridDim.x;

        auto *td_x = tma_xysfa + group * 3;
        auto *td_sfa = td_x + 2;
        tma_descriptor_fence_acquire(td_x);
        tma_descriptor_fence_acquire(td_sfa);

        auto gX = gX_mkl(_, _, itile_m, _, _0{});
        auto gW = gW_nkl(_, _, itile_n, _, group);
        auto gSFA = gSFA_mkl(_, _, itile_m, _, _0{});
        auto gSFB = gSFB_nkl(_, _, itile_n, _, group);

        auto tXgX = btma_x.partition_S(gX);
        auto tWgW = btma_w.partition_S(gW);
        auto tSFAgSFA = btma_sfa.partition_S(gSFA);
        auto tSFBgSFB = btma_sfb.partition_S(gSFB);

#pragma unroll 1
        for (int itile_k = 0; itile_k < num_tile_k; ++itile_k) {
          wait_barrier(writable[write_stage], write_phase);
          set_barrier_transaction_bytes(readable[write_stage],
                                        Config::kTmaTransactionBytes);

          cute::copy(tma_x.with(td_x, readable[write_stage]),
                     tXgX(_, _, _, itile_k),
                     tXsX(_, _, _, write_stage));
          cute::copy(tma_w.with(td_w, readable[write_stage]),
                     tWgW(_, _, _, itile_k),
                     tWsW(_, _, _, write_stage));
          cute::copy(tma_sfa.with(td_sfa, readable[write_stage]),
                     tSFAgSFA(_, _, _, itile_k),
                     tSFAsSFA(_, _, _, write_stage));
          cute::copy(tma_sfb.with(td_sfb, readable[write_stage]),
                     tSFBgSFB(_, _, _, itile_k),
                     tSFBsSFB(_, _, _, write_stage));

          ++write_stage;
          if (write_stage == kStage) {
            write_stage = 0;
            write_phase ^= 1;
          }
        }
      }
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<168>();

    TiledMma tiled_mma;
    CollectiveMainloop collective_mainloop;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

    auto tCrX = thread_mma.partition_fragment_A(sX(_, _, _0{}));
    auto tCrW = thread_mma.partition_fragment_B(sW(_, _, _0{}));
    auto tCrSFA = collective_mainloop.partition_fragment_SFA(
        sSFA(_, _, _0{}), thread_mma);
    auto tCrSFB = collective_mainloop.partition_fragment_SFB(
        sSFB(_, _, _0{}), thread_mma);

    auto smem_tiled_copy_x = make_tiled_copy_A(
        typename Config::SmemCopyAtomX{}, tiled_mma);
    auto smem_thr_copy_x = smem_tiled_copy_x.get_thread_slice(thread_idx);
    auto tXsX = smem_thr_copy_x.partition_S(
        as_position_independent_swizzle_tensor(sX));
    auto tXrX = smem_thr_copy_x.retile_D(tCrX);

    auto smem_tiled_copy_w = make_tiled_copy_B(
        typename Config::SmemCopyAtomW{}, tiled_mma);
    auto smem_thr_copy_w = smem_tiled_copy_w.get_thread_slice(thread_idx);
    auto tWsW = smem_thr_copy_w.partition_S(
        as_position_independent_swizzle_tensor(sW));
    auto tWrW = smem_thr_copy_w.retile_D(tCrW);

    auto tile_shape_mnk = tile_shape(tiled_mma);
    auto smem_tiled_copy_sfa = make_tiled_copy_impl(
        typename Config::SmemCopyAtomSFA{},
        collective_mainloop.get_layoutSFA_TV(tiled_mma),
        make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto smem_thr_copy_sfa = smem_tiled_copy_sfa.get_thread_slice(thread_idx);
    auto tSFAsSFA = smem_thr_copy_sfa.partition_S(
        as_position_independent_swizzle_tensor(sSFA));
    auto tSFArSFA = smem_thr_copy_sfa.retile_D(tCrSFA);

    auto smem_tiled_copy_sfb = make_tiled_copy_impl(
        typename Config::SmemCopyAtomSFB{},
        collective_mainloop.get_layoutSFB_TV(tiled_mma),
        make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto smem_thr_copy_sfb = smem_tiled_copy_sfb.get_thread_slice(thread_idx);
    auto tSFBsSFB = smem_thr_copy_sfb.partition_S(
        as_position_independent_swizzle_tensor(sSFB));
    auto tSFBrSFB = smem_thr_copy_sfb.retile_D(tCrSFB);

    auto gC = make_tensor(
        make_gmem_ptr(static_cast<Tout *>(nullptr)),
        make_shape(Int<kTileM>{}, Int<kTileN>{}),
        make_stride(Int<kTileN>{}, _1{}));
    auto accum = thread_mma.partition_fragment_C(gC);
    auto identity_c = make_identity_tensor(shape(gC));
    auto tC_coord = thread_mma.partition_C(identity_c);

    int read_stage = 0;
    int read_phase = 0;
    int iblock = blockIdx.x;
    int group = 0;
    int sum_tile_m = 0;
    int itile_m = 0;
    int itile_n = 0;
    int num_tile_k = k / kTileK + (k % kTileK != 0);

    while (true) {
      get_next_nvfp4_tile_horizon(
          shm_tiles, iblock, num_group, group,
          itile_m, itile_n, sum_tile_m, flat_divider);
      if (group < 0) {
        break;
      }
      iblock += gridDim.x;
      clear(accum);

#pragma unroll 1
      for (int itile_k = 0; itile_k < num_tile_k; ++itile_k) {
        wait_barrier(readable[read_stage], read_phase);

        for_each(make_int_sequence<size<2>(tCrX)>{}, [&](auto k_block) {
          cute::copy(smem_tiled_copy_x,
                     tXsX(_, _, k_block, read_stage),
                     tXrX(_, _, k_block));
          cute::copy(smem_tiled_copy_w,
                     tWsW(_, _, k_block, read_stage),
                     tWrW(_, _, k_block));

          using MMAOp = typename TiledMma::MMA_Op;
          fp4_shift_A(MMAOp{}, tXrX(_, _, k_block));
          fp4_shift_B(MMAOp{}, tWrW(_, _, k_block));

          cute::copy(tSFAsSFA(_, _, k_block, read_stage),
                     tSFArSFA(_, _, k_block));
          cute::copy(tSFBsSFB(_, _, k_block, read_stage),
                     tSFBrSFB(_, _, k_block));

          cute::gemm(
              tiled_mma,
              make_zip_tensor(tCrX(_, _, k_block),
                              tCrSFA(_, _, k_block)),
              make_zip_tensor(tCrW(_, _, k_block),
                              tCrSFB(_, _, k_block)),
              accum);
        });

        arrive_barrier(writable[read_stage]);
        ++read_stage;
        if (read_stage == kStage) {
          read_stage = 0;
          read_phase ^= 1;
        }
      }

#pragma unroll
      for (int i = 0; i < size(accum); ++i) {
        auto coord = tC_coord(i);
        int row = itile_m * kTileM + get<0>(coord);
        int col = itile_n * kTileN + get<1>(coord);
        if (row < seqlens_ptr[group] && col < n) {
          int64_t output_offset =
              (int64_t(cu_seqlens_ptr[group]) + row) * n + col;
          y_ptr[output_offset] = static_cast<Tout>(accum(i));
        }
      }
    }
  }
}

}  // namespace kernels
}  // namespace grouped_gemm
}  // namespace sm120_nvfp4

#endif  // SM120_NVFP4_GROUPED_GEMM_KERNELS_CUH_
