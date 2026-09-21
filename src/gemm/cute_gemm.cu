#include "sm120_nvfp4/gemm.hpp"

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include <cuda_runtime.h>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "common/gemm_config.cuh"

namespace sm120_nvfp4 {
namespace cute_gemm_detail {

using namespace cute;  // NOLINT

constexpr int kTmaStoreMinRows = 64;

template <typename Config, typename TmaOutput, bool kUseTmaStore>
__global__ void __launch_bounds__(384, 1) nvfp4_gemm_kernel(
    CUTLASS_GRID_CONSTANT typename Config::TmaX const tma_x,
    CUTLASS_GRID_CONSTANT typename Config::TmaW const tma_w,
    CUTLASS_GRID_CONSTANT typename Config::TmaSFA const tma_sfa,
    CUTLASS_GRID_CONSTANT typename Config::TmaSFB const tma_sfb,
    CUTLASS_GRID_CONSTANT TmaOutput const tma_output,
    typename Config::Tout* output, int groups, int m, int n, int k) {
  using Tin = typename Config::Tin;
  using Tout = typename Config::Tout;
  using TS = typename Config::TS;
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
  constexpr int kStages = Config::kStage;
  constexpr int kMathThreads = size(TiledMma{});
  using OutputSmemLayout = Layout<
      Shape<Int<kTileM>, Int<kTileN>>,
      Stride<Int<kTileN>, _1>>;
  constexpr int kOutputSmemOffset =
      (sizeof(TensorStorage) + 127) / 128 * 128;
  static_assert(kMathThreads == 256,
                "SM120 cooperative NVFP4 MMA requires 256 math threads");

  int thread_idx = threadIdx.x;
  int elected = cute::elect_one_sync();

  __shared__ uint64_t readable[kStages];
  __shared__ uint64_t writable[kStages];
  extern __shared__ uint8_t shared_memory[] alignas(1024);
  TensorStorage& storage =
      *reinterpret_cast<TensorStorage*>(shared_memory);

  auto sX = make_tensor(make_smem_ptr(storage.smem_X.begin()), SLayoutX{});
  auto sW = make_tensor(make_smem_ptr(storage.smem_W.begin()), SLayoutW{});
  auto sSFA =
      make_tensor(make_smem_ptr(storage.smem_SFA.begin()), SLayoutSFA{});
  auto sSFB =
      make_tensor(make_smem_ptr(storage.smem_SFB.begin()), SLayoutSFB{});

  auto layout_sfa = Config::make_layout_sfa(m, n, k, groups);
  auto layout_sfb = Config::make_layout_sfb(m, n, k, groups);
  auto mX = tma_x.get_tma_tensor(make_shape(m, k, groups));
  auto mW = tma_w.get_tma_tensor(make_shape(n, k, groups));
  auto mSFA = tma_sfa.get_tma_tensor(shape(layout_sfa));
  auto mSFB = tma_sfb.get_tma_tensor(shape(layout_sfb));

  using X = Underscore;
  using TileShape = typename Config::TileShape;
  auto gX_mkl = local_tile(
      mX, TileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});
  auto gW_nkl = local_tile(
      mW, TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});
  auto gSFA_mkl = local_tile(
      mSFA, TileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});
  auto gSFB_nkl = local_tile(
      mSFB, TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});

  auto tma_x_slice = tma_x.get_slice(0);
  auto tma_w_slice = tma_w.get_slice(0);
  auto tma_sfa_slice = tma_sfa.get_slice(0);
  auto tma_sfb_slice = tma_sfb.get_slice(0);
  auto tXsX = tma_x_slice.partition_D(sX);
  auto tWsW = tma_w_slice.partition_D(sW);
  auto tSFAsSFA = tma_sfa_slice.partition_D(sSFA);
  auto tSFBsSFB = tma_sfb_slice.partition_D(sSFB);

  if (thread_idx < kStages) {
    initialize_barrier(readable[thread_idx], 1);
    initialize_barrier(writable[thread_idx], kMathThreads);
  }
  __syncthreads();

  int tile_count_m = (m + kTileM - 1) / kTileM;
  int tile_count_n = (n + kTileN - 1) / kTileN;
  int tile_count_k = (k + kTileK - 1) / kTileK;
  int tiles_per_group = tile_count_m * tile_count_n;
  int output_tile_count = groups * tiles_per_group;

  if (thread_idx >= kMathThreads) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    int load_thread = thread_idx - kMathThreads;
    int load_warp = __shfl_sync(0xffffffff, load_thread / 32, 0);
    bool is_load_leader = load_warp == 0 && elected;

    if (is_load_leader) {
      int write_stage = 0;
      int write_phase = 1;

      for (int flat_tile = blockIdx.x; flat_tile < output_tile_count;
           flat_tile += gridDim.x) {
        int group = flat_tile / tiles_per_group;
        int local_tile = flat_tile % tiles_per_group;
        int tile_m = local_tile / tile_count_n;
        int tile_n = local_tile % tile_count_n;

        auto gX = gX_mkl(_, _, tile_m, _, group);
        auto gW = gW_nkl(_, _, tile_n, _, group);
        auto gSFA = gSFA_mkl(_, _, tile_m, _, group);
        auto gSFB = gSFB_nkl(_, _, tile_n, _, group);
        auto tXgX = tma_x_slice.partition_S(gX);
        auto tWgW = tma_w_slice.partition_S(gW);
        auto tSFAgSFA = tma_sfa_slice.partition_S(gSFA);
        auto tSFBgSFB = tma_sfb_slice.partition_S(gSFB);

#pragma unroll 1
        for (int tile_k = 0; tile_k < tile_count_k; ++tile_k) {
          wait_barrier(writable[write_stage], write_phase);
          set_barrier_transaction_bytes(
              readable[write_stage], Config::kTmaTransactionBytes);

          cute::copy(tma_x.with(readable[write_stage]),
                     tXgX(_, _, _, tile_k),
                     tXsX(_, _, _, write_stage));
          cute::copy(tma_w.with(readable[write_stage]),
                     tWgW(_, _, _, tile_k),
                     tWsW(_, _, _, write_stage));
          cute::copy(tma_sfa.with(readable[write_stage]),
                     tSFAgSFA(_, _, _, tile_k),
                     tSFAsSFA(_, _, _, write_stage));
          cute::copy(tma_sfb.with(readable[write_stage]),
                     tSFBgSFB(_, _, _, tile_k),
                     tSFBsSFB(_, _, _, write_stage));

          ++write_stage;
          if (write_stage == kStages) {
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

    auto smem_tiled_copy_x =
        make_tiled_copy_A(typename Config::SmemCopyAtomX{}, tiled_mma);
    auto smem_thread_copy_x =
        smem_tiled_copy_x.get_thread_slice(thread_idx);
    auto tXsX = smem_thread_copy_x.partition_S(
        as_position_independent_swizzle_tensor(sX));
    auto tXrX = smem_thread_copy_x.retile_D(tCrX);

    auto smem_tiled_copy_w =
        make_tiled_copy_B(typename Config::SmemCopyAtomW{}, tiled_mma);
    auto smem_thread_copy_w =
        smem_tiled_copy_w.get_thread_slice(thread_idx);
    auto tWsW = smem_thread_copy_w.partition_S(
        as_position_independent_swizzle_tensor(sW));
    auto tWrW = smem_thread_copy_w.retile_D(tCrW);

    auto tile_shape_mnk = tile_shape(tiled_mma);
    auto smem_tiled_copy_sfa = make_tiled_copy_impl(
        typename Config::SmemCopyAtomSFA{},
        collective_mainloop.get_layoutSFA_TV(tiled_mma),
        make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto smem_thread_copy_sfa =
        smem_tiled_copy_sfa.get_thread_slice(thread_idx);
    auto tSFAsSFA = smem_thread_copy_sfa.partition_S(
        as_position_independent_swizzle_tensor(sSFA));
    auto tSFArSFA = smem_thread_copy_sfa.retile_D(tCrSFA);

    auto smem_tiled_copy_sfb = make_tiled_copy_impl(
        typename Config::SmemCopyAtomSFB{},
        collective_mainloop.get_layoutSFB_TV(tiled_mma),
        make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto smem_thread_copy_sfb =
        smem_tiled_copy_sfb.get_thread_slice(thread_idx);
    auto tSFBsSFB = smem_thread_copy_sfb.partition_S(
        as_position_independent_swizzle_tensor(sSFB));
    auto tSFBrSFB = smem_thread_copy_sfb.retile_D(tCrSFB);

    auto output_tile = make_tensor(
        make_gmem_ptr(static_cast<Tout*>(nullptr)),
        make_shape(Int<kTileM>{}, Int<kTileN>{}),
        make_stride(Int<kTileN>{}, _1{}));
    auto accum = thread_mma.partition_fragment_C(output_tile);
    auto identity = make_identity_tensor(shape(output_tile));
    auto output_coordinates = thread_mma.partition_C(identity);

    auto output_tma_slice = tma_output.get_slice(0);
    auto output_coordinates_mng = tma_output.get_tma_tensor(
        make_shape(m, n, groups));
    auto shared_output = make_tensor(
        make_smem_ptr(reinterpret_cast<Tout*>(
            shared_memory + kOutputSmemOffset)),
        OutputSmemLayout{});
    auto tma_shared_output = output_tma_slice.partition_S(shared_output);

    int read_stage = 0;
    int read_phase = 0;

    for (int flat_tile = blockIdx.x; flat_tile < output_tile_count;
         flat_tile += gridDim.x) {
      int group = flat_tile / tiles_per_group;
      int local_tile = flat_tile % tiles_per_group;
      int tile_m = local_tile / tile_count_n;
      int tile_n = local_tile % tile_count_n;
      clear(accum);

#pragma unroll 1
      for (int tile_k = 0; tile_k < tile_count_k; ++tile_k) {
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
        if (read_stage == kStages) {
          read_stage = 0;
          read_phase ^= 1;
        }
      }

#pragma unroll
      for (int i = 0; i < size(accum); ++i) {
        auto coordinate = output_coordinates(i);
        if constexpr (kUseTmaStore) {
          shared_output(get<0>(coordinate), get<1>(coordinate)) =
              static_cast<Tout>(accum(i));
        } else {
          int row = tile_m * kTileM + get<0>(coordinate);
          int column = tile_n * kTileN + get<1>(coordinate);
          if (row < m && column < n) {
            output[(static_cast<int64_t>(group) * m + row) * n + column] =
                static_cast<Tout>(accum(i));
          }
        }
      }

      if constexpr (kUseTmaStore) {
        // All MMA threads first stage a dense FP16 tile in shared memory.
        // The producer warpgroup can continue filling the disjoint mainloop
        // buffers while TMA writes the epilogue tile to global memory.
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::NamedBarrier::sync(
            kMathThreads,
            cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        if (thread_idx == 0) {
          auto global_output = cute::local_tile(
              output_coordinates_mng,
              make_shape(Int<kTileM>{}, Int<kTileN>{}),
              make_coord(tile_m, tile_n, group));
          auto tma_global_output =
              output_tma_slice.partition_D(global_output);
          cute::copy(tma_output, tma_shared_output, tma_global_output);
          cute::tma_store_arrive();
          cute::tma_store_wait<0>();
        }
        // A single shared tile is reused across persistent work items, so do
        // not let any consumer overwrite it before the TMA store completes.
        cutlass::arch::NamedBarrier::sync(
            kMathThreads,
            cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
      }
    }
  }
}

template <typename Config, typename TmaOutput, bool kUseTmaStore>
GemmStatus launch_kernel_specialization(
    const typename Config::TmaBundle& tma, const TmaOutput& tma_output,
    typename Config::Tout* output, int grid_size,
    int groups, int m, int n, int k, cudaStream_t stream) {
  constexpr int kThreads = 384;
  constexpr int kOutputSmemOffset =
      (Config::get_shm_size() + 127) / 128 * 128;
  constexpr int kSharedMemoryBytes =
      kUseTmaStore
          ? kOutputSmemOffset +
                Config::kTileM * Config::kTileN *
                    sizeof(typename Config::Tout)
          : Config::get_shm_size();
  auto kernel = nvfp4_gemm_kernel<Config, TmaOutput, kUseTmaStore>;
  if (cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
          kSharedMemoryBytes) != cudaSuccess) {
    return GemmStatus::kCudaError;
  }
  kernel<<<grid_size, kThreads, kSharedMemoryBytes, stream>>>(
      tma.x, tma.w, tma.sfa, tma.sfb, tma_output,
      output, groups, m, n, k);
  return cudaPeekAtLastError() == cudaSuccess
             ? GemmStatus::kSuccess
             : GemmStatus::kCudaError;
}

bool valid_shape(int m, int n, int k) {
  return m > 0 && n > 0 && k > 0 &&
         (k % kInputAlignmentElements) == 0 &&
         (n % kOutputAlignmentElements) == 0;
}

template <typename Tout>
GemmStatus launch_nvfp4_cute_gemm_sm120(
    int groups, int m, int n, int k, const void* a, const void* b,
    const void* sfa, const void* sfb, Tout* c, cudaStream_t stream) {
  if (groups <= 0 || !cute_gemm_detail::valid_shape(m, n, k) || a == nullptr ||
      b == nullptr || sfa == nullptr || sfb == nullptr || c == nullptr) {
    return GemmStatus::kInvalidArgument;
  }

  int device = 0;
  cudaDeviceProp properties{};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
    return GemmStatus::kCudaError;
  }
  if (properties.major != 12 || properties.minor != 0) {
    return GemmStatus::kUnsupportedDevice;
  }

  using Config = detail::Nvfp4GemmConfig<Tout, 128, 128, 128, 3>;
  using TmaInternalElementX = typename Config::TmaInternalElementX;
  using TmaInternalElementW = typename Config::TmaInternalElementW;
  using Scale = typename Config::TS;

  Config config;
  auto input = cute::make_tensor(
      cute::recast_ptr<TmaInternalElementX>(a),
      cute::make_shape(int32_t(m), int32_t(k), int32_t(groups)),
      typename Config::StrideX{int64_t(k), cute::Int<1>{},
                               int64_t(m) * k});
  auto weight = cute::make_tensor(
      cute::recast_ptr<TmaInternalElementW>(b),
      cute::make_shape(int32_t(n), int32_t(k), int32_t(groups)),
      typename Config::StrideW{int64_t(k), cute::Int<1>{},
                               int64_t(n) * k});
  auto layout_sfa = Config::make_layout_sfa(m, n, k, groups);
  auto layout_sfb = Config::make_layout_sfb(m, n, k, groups);
  auto input_scale = cute::make_tensor(
      reinterpret_cast<const Scale*>(sfa), layout_sfa);
  auto weight_scale = cute::make_tensor(
      reinterpret_cast<const Scale*>(sfb), layout_sfb);
  auto tma = config.get_tma(input, weight, input_scale, weight_scale);
  using OutputSmemLayout = cute::Layout<
      cute::Shape<cute::Int<Config::kTileM>, cute::Int<Config::kTileN>>,
      cute::Stride<cute::Int<Config::kTileN>, cute::_1>>;
  auto output = cute::make_tensor(
      cute::make_gmem_ptr(c),
      cute::make_shape(int32_t(m), int32_t(n), int32_t(groups)),
      cute::make_stride(int64_t(n), cute::Int<1>{}, int64_t(m) * n));
  auto tma_output = cute::make_tma_copy(
      cute::SM90_TMA_STORE{}, output, OutputSmemLayout{},
      cute::make_shape(cute::Int<Config::kTileM>{},
                       cute::Int<Config::kTileN>{}),
      cute::_1{});

  int tile_count = groups * cute::ceil_div(m, Config::kTileM) *
                   cute::ceil_div(n, Config::kTileN);
  int grid_size = std::min(properties.multiProcessorCount, tile_count);
  constexpr bool kTmaOutputSupported =
      std::is_same_v<Tout, cute::half_t>;
  // Staging a complete 128x128 output tile has a fixed cost. Measurements on
  // SM120 show that TMA wins from 64 valid rows onward, while scalar stores
  // remain faster for the smaller decode-style M shapes.
  bool use_tma_store =
      kTmaOutputSupported && m >= cute_gemm_detail::kTmaStoreMinRows;
  if constexpr (kTmaOutputSupported) {
    if (use_tma_store) {
      return cute_gemm_detail::launch_kernel_specialization<
          Config, decltype(tma_output), true>(
          tma, tma_output, reinterpret_cast<typename Config::Tout*>(c),
          grid_size, groups, m, n, k, stream);
    }
  }
  return cute_gemm_detail::launch_kernel_specialization<
      Config, decltype(tma_output), false>(
      tma, tma_output, reinterpret_cast<typename Config::Tout*>(c),
      grid_size, groups, m, n, k, stream);
}

}  // namespace cute_gemm_detail

GemmStatus nvfp4_cute_gemm_sm120(
    int m, int n, int k, const void* a, const void* b,
    const void* sfa, const void* sfb, half* c, cudaStream_t stream) {
  return cute_gemm_detail::launch_nvfp4_cute_gemm_sm120(
      1, m, n, k, a, b, sfa, sfb,
      reinterpret_cast<cute::half_t*>(c), stream);
}

GemmStatus nvfp4_cute_gemm_f32_sm120(
    int m, int n, int k, const void* a, const void* b,
    const void* sfa, const void* sfb, float* c, cudaStream_t stream) {
  return cute_gemm_detail::launch_nvfp4_cute_gemm_sm120(
      1, m, n, k, a, b, sfa, sfb, c, stream);
}

GemmStatus nvfp4_cute_batched_gemm_sm120(
    int groups, int m, int n, int k, const void* a, const void* b,
    const void* sfa, const void* sfb, half* c, cudaStream_t stream) {
  return cute_gemm_detail::launch_nvfp4_cute_gemm_sm120(
      groups, m, n, k, a, b, sfa, sfb,
      reinterpret_cast<cute::half_t*>(c), stream);
}

GemmStatus nvfp4_cute_batched_gemm_f32_sm120(
    int groups, int m, int n, int k, const void* a, const void* b,
    const void* sfa, const void* sfb, float* c, cudaStream_t stream) {
  return cute_gemm_detail::launch_nvfp4_cute_gemm_sm120(
      groups, m, n, k, a, b, sfa, sfb, c, stream);
}

}  // namespace sm120_nvfp4
