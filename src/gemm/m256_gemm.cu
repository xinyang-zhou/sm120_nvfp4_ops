#include "gemm/m256_gemm.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cute/atom/copy_traits_sm90.hpp"
#include "cute/atom/mma_traits_sm90_gmma.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "common/gemm_config.cuh"

namespace sm120_nvfp4::m256_experiment {
namespace {

using namespace cute;  // NOLINT

constexpr int kTileM = 128;
constexpr int kTileN = 128;
constexpr int kTileK = 128;
constexpr int kStages = 3;
constexpr int kThreads = 384;
constexpr int kReductionThreads = 256;
constexpr int kPartialStoreRows = 64;
constexpr int kSpecializedM = 256;
constexpr int kSpecializedSplitK = 2;

using Config = detail::Nvfp4GemmConfig<float, kTileM, kTileN, kTileK, kStages>;
using PartialOutputSmemLayout = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<float>{},
    Shape<Int<kPartialStoreRows>, Int<kTileN>>{}));

static_assert(size(PartialOutputSmemLayout{}) ==
                  kPartialStoreRows * kTileN,
              "The swizzled partial-output layout must cover one store slab");

template <typename GemmConfig, typename TmaPartialOutput>
__global__ void __launch_bounds__(kThreads, 1) nvfp4_m256_partial_kernel(
    CUTLASS_GRID_CONSTANT typename GemmConfig::TmaX const tma_x,
    CUTLASS_GRID_CONSTANT typename GemmConfig::TmaW const tma_w,
    CUTLASS_GRID_CONSTANT typename GemmConfig::TmaSFA const tma_sfa,
    CUTLASS_GRID_CONSTANT typename GemmConfig::TmaSFB const tma_sfb,
    CUTLASS_GRID_CONSTANT TmaPartialOutput const tma_partial_output,
    int groups, int m, int n, int k, int split_k) {
  using TiledMma = typename GemmConfig::TiledMma;
  using CollectiveMainloop = typename GemmConfig::CollectiveMainloop;
  using TensorStorage = typename GemmConfig::TensorStorage;
  using SLayoutX = typename GemmConfig::SLayoutX;
  using SLayoutW = typename GemmConfig::SLayoutW;
  using SLayoutSFA = typename GemmConfig::SLayoutSFA;
  using SLayoutSFB = typename GemmConfig::SLayoutSFB;

  constexpr int kConfigTileM = GemmConfig::kTileM;
  constexpr int kConfigTileN = GemmConfig::kTileN;
  constexpr int kConfigTileK = GemmConfig::kTileK;
  constexpr int kConfigStages = GemmConfig::kStage;
  constexpr int kMathThreads = size(TiledMma{});
  constexpr int kPartialStoreCount =
      kConfigTileM / kPartialStoreRows;
  constexpr int kPartialOutputSmemOffset =
      (sizeof(TensorStorage) + 127) / 128 * 128;
  static_assert(kMathThreads == 256,
                "SM120 cooperative NVFP4 MMA requires 256 math threads");
  static_assert(kPartialOutputSmemOffset % 1024 == 0,
                "The SW128 staging buffer must be 1024-byte aligned");
  static_assert(kConfigTileN == kTileN,
                "The partial-output shared layout is specialized for tile N=128");
  static_assert(kConfigTileM % kPartialStoreRows == 0,
                "The partial-output tile must split evenly into TMA stores");

  const int thread_idx = threadIdx.x;
  const int elected = cute::elect_one_sync();

  __shared__ uint64_t readable[kConfigStages];
  __shared__ uint64_t writable[kConfigStages];
  extern __shared__ uint8_t shared_memory[] alignas(1024);
  TensorStorage& storage =
      *reinterpret_cast<TensorStorage*>(shared_memory);

  auto sX = make_tensor(make_smem_ptr(storage.smem_X.begin()), SLayoutX{});
  auto sW = make_tensor(make_smem_ptr(storage.smem_W.begin()), SLayoutW{});
  auto sSFA =
      make_tensor(make_smem_ptr(storage.smem_SFA.begin()), SLayoutSFA{});
  auto sSFB =
      make_tensor(make_smem_ptr(storage.smem_SFB.begin()), SLayoutSFB{});

  auto layout_sfa = GemmConfig::make_layout_sfa(m, n, k, groups);
  auto layout_sfb = GemmConfig::make_layout_sfb(m, n, k, groups);
  auto mX = tma_x.get_tma_tensor(make_shape(m, k, groups));
  auto mW = tma_w.get_tma_tensor(make_shape(n, k, groups));
  auto mSFA = tma_sfa.get_tma_tensor(shape(layout_sfa));
  auto mSFB = tma_sfb.get_tma_tensor(shape(layout_sfb));

  using X = Underscore;
  using TileShape = typename GemmConfig::TileShape;
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

  if (thread_idx < kConfigStages) {
    initialize_barrier(readable[thread_idx], 1);
    initialize_barrier(writable[thread_idx], kMathThreads);
  }
  __syncthreads();

  const int tile_count_m = (m + kConfigTileM - 1) / kConfigTileM;
  const int tile_count_n = (n + kConfigTileN - 1) / kConfigTileN;
  const int tile_count_k = (k + kConfigTileK - 1) / kConfigTileK;
  const int tiles_per_group = tile_count_m * tile_count_n;
  const int output_tile_count = groups * tiles_per_group;
  const int task_count = output_tile_count * split_k;

  if (thread_idx >= kMathThreads) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    const int load_thread = thread_idx - kMathThreads;
    const int load_warp = __shfl_sync(0xffffffff, load_thread / 32, 0);
    const bool is_load_leader = load_warp == 0 && elected;

    if (is_load_leader) {
      int write_stage = 0;
      int write_phase = 1;

      for (int flat_task = blockIdx.x; flat_task < task_count;
           flat_task += gridDim.x) {
        const int flat_tile = flat_task / split_k;
        const int split = flat_task % split_k;
        const int group = flat_tile / tiles_per_group;
        const int local_tile = flat_tile % tiles_per_group;
        const int tile_m = local_tile / tile_count_n;
        const int tile_n = local_tile % tile_count_n;
        const int tile_k_begin = tile_count_k * split / split_k;
        const int tile_k_end = tile_count_k * (split + 1) / split_k;

        auto gX = gX_mkl(_, _, tile_m, _, group);
        auto gW = gW_nkl(_, _, tile_n, _, group);
        auto gSFA = gSFA_mkl(_, _, tile_m, _, group);
        auto gSFB = gSFB_nkl(_, _, tile_n, _, group);
        auto tXgX = tma_x_slice.partition_S(gX);
        auto tWgW = tma_w_slice.partition_S(gW);
        auto tSFAgSFA = tma_sfa_slice.partition_S(gSFA);
        auto tSFBgSFB = tma_sfb_slice.partition_S(gSFB);

#pragma unroll 1
        for (int tile_k = tile_k_begin; tile_k < tile_k_end; ++tile_k) {
          wait_barrier(writable[write_stage], write_phase);
          set_barrier_transaction_bytes(
              readable[write_stage], GemmConfig::kTmaTransactionBytes);

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
          if (write_stage == kConfigStages) {
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
        make_tiled_copy_A(typename GemmConfig::SmemCopyAtomX{}, tiled_mma);
    auto smem_thread_copy_x =
        smem_tiled_copy_x.get_thread_slice(thread_idx);
    auto tXsX = smem_thread_copy_x.partition_S(
        as_position_independent_swizzle_tensor(sX));
    auto tXrX = smem_thread_copy_x.retile_D(tCrX);

    auto smem_tiled_copy_w =
        make_tiled_copy_B(typename GemmConfig::SmemCopyAtomW{}, tiled_mma);
    auto smem_thread_copy_w =
        smem_tiled_copy_w.get_thread_slice(thread_idx);
    auto tWsW = smem_thread_copy_w.partition_S(
        as_position_independent_swizzle_tensor(sW));
    auto tWrW = smem_thread_copy_w.retile_D(tCrW);

    auto tile_shape_mnk = tile_shape(tiled_mma);
    auto smem_tiled_copy_sfa = make_tiled_copy_impl(
        typename GemmConfig::SmemCopyAtomSFA{},
        collective_mainloop.get_layoutSFA_TV(tiled_mma),
        make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto smem_thread_copy_sfa =
        smem_tiled_copy_sfa.get_thread_slice(thread_idx);
    auto tSFAsSFA = smem_thread_copy_sfa.partition_S(
        as_position_independent_swizzle_tensor(sSFA));
    auto tSFArSFA = smem_thread_copy_sfa.retile_D(tCrSFA);

    auto smem_tiled_copy_sfb = make_tiled_copy_impl(
        typename GemmConfig::SmemCopyAtomSFB{},
        collective_mainloop.get_layoutSFB_TV(tiled_mma),
        make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto smem_thread_copy_sfb =
        smem_tiled_copy_sfb.get_thread_slice(thread_idx);
    auto tSFBsSFB = smem_thread_copy_sfb.partition_S(
        as_position_independent_swizzle_tensor(sSFB));
    auto tSFBrSFB = smem_thread_copy_sfb.retile_D(tCrSFB);

    auto output_tile = make_tensor(
        make_gmem_ptr(static_cast<float*>(nullptr)),
        make_shape(Int<kConfigTileM>{}, Int<kConfigTileN>{}),
        make_stride(Int<kConfigTileN>{}, _1{}));
    auto accum = thread_mma.partition_fragment_C(output_tile);

    auto partial_output_tma_slice = tma_partial_output.get_slice(0);
    auto partial_output_coordinates = tma_partial_output.get_tma_tensor(
        make_shape(m, n, groups * split_k));
    auto shared_partial_output = as_position_independent_swizzle_tensor(
        make_tensor(
            make_smem_ptr(reinterpret_cast<float*>(
                shared_memory + kPartialOutputSmemOffset)),
            PartialOutputSmemLayout{}));
    auto tma_shared_partial_output =
        partial_output_tma_slice.partition_S(shared_partial_output);

    // Follow the CUTLASS SM120 epilogue's accumulator-to-shared tiling.  The
    // STSM atom is used only as the reference thread/value layout; FP32 data
    // is stored by the auto-vectorizing copy selected for FP32 accumulators.
    using ReferenceCopyAtom =
        Copy_Atom<SM90_U32x2_STSM_N, cutlass::half_t>;
    auto tiled_copy_c_atom =
        make_tiled_copy_C_atom(ReferenceCopyAtom{}, tiled_mma);
    auto tiled_r2s = make_tiled_copy_S(
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, float>{},
        tiled_copy_c_atom);
    auto thread_r2s = tiled_r2s.get_thread_slice(thread_idx);
    auto tRS_rAcc = thread_r2s.retile_S(accum);
    auto tRS_sPartialOutput =
        thread_r2s.partition_D(shared_partial_output);
    static_assert(decltype(size<1>(tRS_rAcc))::value ==
                      kPartialStoreCount,
                  "The R2S accumulator partition must expose one mode per slab");
    static_assert(
        decltype(max_common_vector(
            layout(tRS_rAcc(_, _0{}, _)),
            layout(tRS_sPartialOutput)))::value >= 2,
        "The FP32 partial-output store must retain at least 64-bit vectors");

    int read_stage = 0;
    int read_phase = 0;

    for (int flat_task = blockIdx.x; flat_task < task_count;
         flat_task += gridDim.x) {
      const int flat_tile = flat_task / split_k;
      const int split = flat_task % split_k;
      const int group = flat_tile / tiles_per_group;
      const int local_tile = flat_tile % tiles_per_group;
      const int tile_m = local_tile / tile_count_n;
      const int tile_n = local_tile % tile_count_n;
      const int tile_k_begin = tile_count_k * split / split_k;
      const int tile_k_end = tile_count_k * (split + 1) / split_k;
      clear(accum);

#pragma unroll 1
      for (int tile_k = tile_k_begin; tile_k < tile_k_end; ++tile_k) {
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
        if (read_stage == kConfigStages) {
          read_stage = 0;
          read_phase ^= 1;
        }
      }

      // A full FP32 128x128 tile would require 64 KiB of staging storage.
      // Stage two 64-row slabs through the same 32 KiB buffer instead.
      for_each(make_int_sequence<kPartialStoreCount>{}, [&](auto store) {
        copy(tiled_r2s, tRS_rAcc(_, store, _), tRS_sPartialOutput);

        cutlass::arch::fence_view_async_shared();
        cutlass::arch::NamedBarrier::sync(
            kMathThreads,
            cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        if (thread_idx == 0) {
          auto global_partial_output = cute::local_tile(
              partial_output_coordinates,
              make_shape(Int<kPartialStoreRows>{},
                         Int<kConfigTileN>{}),
              make_coord(
                  tile_m * kPartialStoreCount + store,
                  tile_n, group + split * groups));
          auto tma_global_partial_output =
              partial_output_tma_slice.partition_D(global_partial_output);
          cute::copy(
              tma_partial_output, tma_shared_partial_output,
              tma_global_partial_output);
          cute::tma_store_arrive();
          cute::tma_store_wait<0>();
        }
        cutlass::arch::NamedBarrier::sync(
            kMathThreads,
            cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
      });
    }
  }
}

// The M=256 path is fixed to split_k=2.  Spell out both loads so the
// compiler can issue them independently before their dependent FP32 add.
__global__ void reduce_m256_split2_kernel(
    const float* partial_output, half* output,
    int64_t output_elements) {
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < output_elements;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const float partial_0 = partial_output[index];
    const float partial_1 = partial_output[output_elements + index];
    output[index] = __float2half_rn(partial_0 + partial_1);
  }
}

bool valid_shape(int groups, int m, int n, int k, int split_k) {
  if (groups <= 0 || m != kSpecializedM || n <= 0 || k <= 0 ||
      split_k != kSpecializedSplitK ||
      (k % kInputAlignmentElements) != 0 ||
      (n % kOutputAlignmentElements) != 0) {
    return false;
  }
  const int tile_count_k = (k + kTileK - 1) / kTileK;
  return split_k <= tile_count_k;
}

}  // namespace

std::size_t workspace_size(
    int groups, int m, int n, int k, int split_k) {
  if (!valid_shape(groups, m, n, k, split_k)) {
    return 0;
  }

  constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
  std::size_t elements = static_cast<std::size_t>(groups);
  for (int extent : {m, n, split_k}) {
    if (elements > kMax / static_cast<std::size_t>(extent)) {
      return 0;
    }
    elements *= static_cast<std::size_t>(extent);
  }
  if (elements > kMax / sizeof(float)) {
    return 0;
  }
  return elements * sizeof(float);
}

GemmStatus launch(
    int groups, int m, int n, int k, int split_k,
    const void* a, const void* b,
    const void* sfa, const void* sfb,
    half* output,
    void* workspace, std::size_t workspace_bytes,
    cudaStream_t stream) {
  const std::size_t required_workspace =
      workspace_size(groups, m, n, k, split_k);
  if (required_workspace == 0 || a == nullptr || b == nullptr ||
      sfa == nullptr || sfb == nullptr || output == nullptr ||
      workspace == nullptr ||
      (reinterpret_cast<std::uintptr_t>(workspace) % alignof(float)) != 0) {
    return GemmStatus::kInvalidArgument;
  }
  if (workspace_bytes < required_workspace) {
    return GemmStatus::kInsufficientWorkspace;
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

  Config config;
  auto input = cute::make_tensor(
      cute::recast_ptr<typename Config::TmaInternalElementX>(a),
      cute::make_shape(int32_t(m), int32_t(k), int32_t(groups)),
      typename Config::StrideX{
          int64_t(k), cute::Int<1>{}, int64_t(m) * k});
  auto weight = cute::make_tensor(
      cute::recast_ptr<typename Config::TmaInternalElementW>(b),
      cute::make_shape(int32_t(n), int32_t(k), int32_t(groups)),
      typename Config::StrideW{
          int64_t(k), cute::Int<1>{}, int64_t(n) * k});
  auto layout_sfa = Config::make_layout_sfa(m, n, k, groups);
  auto layout_sfb = Config::make_layout_sfb(m, n, k, groups);
  auto input_scale = cute::make_tensor(
      reinterpret_cast<const typename Config::TS*>(sfa), layout_sfa);
  auto weight_scale = cute::make_tensor(
      reinterpret_cast<const typename Config::TS*>(sfb), layout_sfb);
  auto tma = config.get_tma(input, weight, input_scale, weight_scale);
  // Flatten (split, group) into one batch coordinate while preserving the
  // workspace layout [split][group][m][n].
  auto partial_output = cute::make_tensor(
      cute::make_gmem_ptr(static_cast<float*>(workspace)),
      cute::make_shape(
          int32_t(m), int32_t(n), int32_t(groups * split_k)),
      cute::make_stride(
          int64_t(n), cute::Int<1>{}, int64_t(m) * n));
  auto tma_partial_output = cute::make_tma_copy(
      cute::SM90_TMA_STORE{}, partial_output,
      PartialOutputSmemLayout{},
      cute::make_shape(cute::Int<kPartialStoreRows>{},
                       cute::Int<Config::kTileN>{}),
      cute::_1{});

  const int tile_count =
      groups * cute::ceil_div(m, Config::kTileM) *
      cute::ceil_div(n, Config::kTileN);
  const int task_count = tile_count * split_k;
  const int grid_size = std::min(properties.multiProcessorCount, task_count);
  auto partial_kernel =
      nvfp4_m256_partial_kernel<Config, decltype(tma_partial_output)>;
  constexpr int kPartialOutputSmemOffset =
      (Config::get_shm_size() + 127) / 128 * 128;
  constexpr int kSharedMemoryBytes =
      kPartialOutputSmemOffset +
      kPartialStoreRows * Config::kTileN * sizeof(float);
  if (cudaFuncSetAttribute(
          partial_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
          kSharedMemoryBytes) != cudaSuccess) {
    return GemmStatus::kCudaError;
  }

  partial_kernel<<<grid_size, kThreads, kSharedMemoryBytes, stream>>>(
      tma.x, tma.w, tma.sfa, tma.sfb, tma_partial_output,
      groups, m, n, k, split_k);
  if (cudaPeekAtLastError() != cudaSuccess) {
    return GemmStatus::kCudaError;
  }

  const int64_t output_elements =
      static_cast<int64_t>(groups) * m * n;
  const int64_t required_reduction_blocks =
      (output_elements + kReductionThreads - 1) / kReductionThreads;
  const int reduction_blocks = static_cast<int>(std::min<int64_t>(
      required_reduction_blocks,
      static_cast<int64_t>(properties.multiProcessorCount) * 4));
  reduce_m256_split2_kernel<<<
      reduction_blocks, kReductionThreads, 0, stream>>>(
      static_cast<const float*>(workspace), output, output_elements);
  return cudaPeekAtLastError() == cudaSuccess
             ? GemmStatus::kSuccess
             : GemmStatus::kCudaError;
}

}  // namespace sm120_nvfp4::m256_experiment
