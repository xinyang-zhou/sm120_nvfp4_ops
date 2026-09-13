#include "sm120_nvfp4/attention.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <cuda_fp4.h>
#include <cuda_runtime.h>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/float8.h"
#include "common/gemm_config.cuh"
#include "fused_moe/scale_layout.cuh"

namespace sm120_nvfp4 {
namespace attention_detail {

constexpr std::size_t kWorkspaceAlignment = 256;
constexpr int kSoftmaxThreads = 256;
constexpr int kDecodeTile = 128;
constexpr int kDecodeTargetCtas = 128;

constexpr std::size_t align_workspace(std::size_t value) {
  return (value + kWorkspaceAlignment - 1) /
         kWorkspaceAlignment * kWorkspaceAlignment;
}

__global__ void softmax_quantize_nvfp4_kernel(
    const float* logits, std::uint8_t* probabilities,
    cutlass::float_ue4m3_t* probability_scale,
    float* row_correction,
    int matrices, int query_length, int kv_length,
    int heads, const int* kv_lengths,
    int query_scale_rows, std::int64_t scale_stride,
    bool causal, float softmax_scale) {
  __shared__ float reduction[kSoftmaxThreads];

  const std::int64_t total_rows =
      static_cast<std::int64_t>(matrices) * query_length;
  const std::int64_t logits_matrix_stride =
      static_cast<std::int64_t>(query_length) * kv_length;
  const std::int64_t probability_matrix_stride = logits_matrix_stride / 2;

  for (std::int64_t linear_row = blockIdx.x;
       linear_row < total_rows; linear_row += gridDim.x) {
    int matrix = static_cast<int>(linear_row / query_length);
    int row = static_cast<int>(linear_row % query_length);
    int last_visible;
    if (kv_lengths != nullptr) {
      int batch = matrix / heads;
      int valid_length = kv_lengths[batch];
      valid_length = valid_length < 0 ? 0 : valid_length;
      valid_length = valid_length > kv_length ? kv_length : valid_length;
      last_visible = valid_length - 1;
    } else {
      last_visible = causal ? kv_length - query_length + row
                            : kv_length - 1;
    }
    const float* logits_row =
        logits + static_cast<std::int64_t>(matrix) * logits_matrix_stride +
        static_cast<std::int64_t>(row) * kv_length;
    std::uint8_t* probability_row =
        probabilities +
        static_cast<std::int64_t>(matrix) * probability_matrix_stride +
        static_cast<std::int64_t>(row) * kv_length / 2;

    float thread_max = -FLT_MAX;
    for (int column = threadIdx.x; column < kv_length;
         column += blockDim.x) {
      if (column <= last_visible) {
        thread_max = fmaxf(thread_max, logits_row[column] * softmax_scale);
      }
    }
    reduction[threadIdx.x] = thread_max;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
      if (threadIdx.x < offset) {
        reduction[threadIdx.x] =
            fmaxf(reduction[threadIdx.x],
                  reduction[threadIdx.x + offset]);
      }
      __syncthreads();
    }
    float row_max = reduction[0];

    float thread_sum = 0.0f;
    for (int column = threadIdx.x; column < kv_length;
         column += blockDim.x) {
      if (column <= last_visible) {
        thread_sum +=
            expf(logits_row[column] * softmax_scale - row_max);
      }
    }
    reduction[threadIdx.x] = thread_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
      if (threadIdx.x < offset) {
        reduction[threadIdx.x] += reduction[threadIdx.x + offset];
      }
      __syncthreads();
    }
    float inverse_sum = reduction[0] > 0.0f ? 1.0f / reduction[0] : 0.0f;

    float thread_quantized_sum = 0.0f;
    int scale_blocks = kv_length / kNvfp4ScaleVectorSize;
    for (int k_block = threadIdx.x; k_block < scale_blocks;
         k_block += blockDim.x) {
      int column = k_block * kNvfp4ScaleVectorSize;
      float values[kNvfp4ScaleVectorSize];
      float amax = 0.0f;
#pragma unroll
      for (int i = 0; i < kNvfp4ScaleVectorSize; ++i) {
        int current_column = column + i;
        float value = 0.0f;
        if (current_column <= last_visible) {
          value = expf(logits_row[current_column] * softmax_scale - row_max) *
                  inverse_sum;
        }
        values[i] = value;
        amax = fmaxf(amax, value);
      }

      cutlass::float_ue4m3_t encoded_scale(0.0f);
      float decoded_scale = 1.0f;
      if (amax > 0.0f) {
        encoded_scale = cutlass::float_ue4m3_t(amax / 6.0f);
        decoded_scale = static_cast<float>(encoded_scale);
        if (!(decoded_scale > 0.0f)) {
          encoded_scale.raw() = 1;
          decoded_scale = static_cast<float>(encoded_scale);
        }
      }

#pragma unroll
      for (int i = 0; i < kNvfp4ScaleVectorSize / 2; ++i) {
        float2 pair;
        pair.x = values[i * 2] / decoded_scale;
        pair.y = values[i * 2 + 1] / decoded_scale;
        __nv_fp4x2_e2m1 packed(pair);
        probability_row[column / 2 + i] = packed.__x;
        float2 reconstructed = static_cast<float2>(packed);
        thread_quantized_sum +=
            (reconstructed.x + reconstructed.y) * decoded_scale;
      }

      std::int64_t scale_offset = fused_moe::sfa_offset(
          row, k_block, query_scale_rows, kv_length);
      probability_scale[static_cast<std::int64_t>(matrix) * scale_stride +
                        scale_offset] = encoded_scale;
    }
    reduction[threadIdx.x] = thread_quantized_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
      if (threadIdx.x < offset) {
        reduction[threadIdx.x] += reduction[threadIdx.x + offset];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      row_correction[linear_row] =
          reduction[0] > 0.0f ? 1.0f / reduction[0] : 0.0f;
    }
    __syncthreads();
  }
}

__global__ void apply_row_correction_kernel(
    half* output, const float* row_correction,
    std::int64_t output_elements, int value_dim) {
  for (std::int64_t index =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < output_elements;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    std::int64_t row = index / value_dim;
    output[index] = __float2half(
        __half2float(output[index]) * row_correction[row]);
  }
}

__global__ void repack_decode_query_scales_kernel(
    const std::uint8_t* source, std::uint8_t* destination,
    int groups, int rows_per_group, int head_dim,
    std::int64_t source_stride, std::int64_t destination_stride) {
  const int scale_blocks = head_dim / kNvfp4ScaleVectorSize;
  const std::int64_t total =
      static_cast<std::int64_t>(groups) * rows_per_group * scale_blocks;
  const int destination_rows =
      align_up(rows_per_group, kScaleMNAlignment);

  for (std::int64_t index =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    int k_block = static_cast<int>(index % scale_blocks);
    std::int64_t row_index = index / scale_blocks;
    int row = static_cast<int>(row_index % rows_per_group);
    int group = static_cast<int>(row_index / rows_per_group);
    std::int64_t source_offset = fused_moe::sfa_offset(
        /*row=*/0, k_block, kScaleMNAlignment, head_dim);
    std::int64_t destination_offset = fused_moe::sfa_offset(
        row, k_block, destination_rows, head_dim);
    destination[static_cast<std::int64_t>(group) * destination_stride +
                destination_offset] =
        source[(static_cast<std::int64_t>(group) * rows_per_group + row) *
                   source_stride +
               source_offset];
  }
}

CUTE_DEVICE float warp_reduce_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return value;
}

int decode_split_count(
    int batch, int kv_heads, int rows_per_group, int max_kv_length) {
  int row_tiles = ceil_div(rows_per_group, kDecodeTile);
  std::int64_t tasks = static_cast<std::int64_t>(batch) * kv_heads * row_tiles;
  int sequence_tiles = ceil_div(max_kv_length, kDecodeTile);
  int occupancy_splits = static_cast<int>(std::max<std::int64_t>(
      1, (kDecodeTargetCtas + tasks - 1) / tasks));
  return std::min(sequence_tiles, occupancy_splits);
}

template <typename Config, bool kPaged>
__global__ void __launch_bounds__(kSoftmaxThreads, 1)
fused_decode_nvfp4_kernel(
    CUTLASS_GRID_CONSTANT typename Config::TmaX const tma_query,
    CUTLASS_GRID_CONSTANT typename Config::TmaW const tma_key,
    CUTLASS_GRID_CONSTANT typename Config::TmaSFA const tma_query_scale,
    CUTLASS_GRID_CONSTANT typename Config::TmaSFB const tma_key_scale,
    CUTLASS_GRID_CONSTANT typename Config::TmaW const tma_value,
    CUTLASS_GRID_CONSTANT typename Config::TmaSFB const tma_value_scale,
    const std::uint8_t* paged_key, const std::uint8_t* paged_value,
    const std::uint8_t* paged_key_scale,
    const std::uint8_t* paged_value_scale,
    const int* block_table,
    const int* kv_lengths, float softmax_scale,
    float* partial_output, float* partial_lse, half* output,
    int groups, int kv_heads, int rows_per_group,
    int max_kv_length, int head_dim, int value_dim, int splits,
    int physical_blocks, int max_blocks_per_sequence, int block_size,
    std::int64_t key_scale_stride, std::int64_t value_scale_stride) {
  using namespace cute;  // NOLINT
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
  static_assert(kTileM == kDecodeTile && kTileN == kDecodeTile);
  static_assert(size(TiledMma{}) == kSoftmaxThreads);
  static_assert(sizeof(TensorStorage) <=
                kDecodeTile * kDecodeTile * sizeof(float));

  int thread_idx = threadIdx.x;
  int row_tile_count = ceil_div(rows_per_group, kDecodeTile);
  int task = blockIdx.x;
  int group = task / row_tile_count;
  int row_tile = task % row_tile_count;
  int split = blockIdx.y;
  if (group >= groups) {
    return;
  }
  int row_start = row_tile * kDecodeTile;
  int valid_rows = min(kDecodeTile, rows_per_group - row_start);
  int batch = group / kv_heads;
  int kv_head = group % kv_heads;
  int valid_length = kv_lengths == nullptr
                         ? max_kv_length
                         : max(0, min(max_kv_length, kv_lengths[batch]));

  int sequence_tiles = ceil_div(max_kv_length, kDecodeTile);
  int tiles_per_split = ceil_div(sequence_tiles, splits);
  int tile_begin = split * tiles_per_split;
  int tile_end = min(sequence_tiles, tile_begin + tiles_per_split);
  tile_end = min(tile_end, ceil_div(valid_length, kDecodeTile));

  std::int64_t first_matrix =
      static_cast<std::int64_t>(group) * rows_per_group + row_start;
  std::int64_t local_output_elements =
      static_cast<std::int64_t>(valid_rows) * value_dim;
  for (std::int64_t index = thread_idx; index < local_output_elements;
       index += blockDim.x) {
    int row = static_cast<int>(index / value_dim);
    int column = static_cast<int>(index % value_dim);
    std::int64_t partial_index =
        ((first_matrix + row) * splits + split) * value_dim + column;
    partial_output[partial_index] = 0.0f;
  }

  __shared__ std::uint64_t readable;
  __shared__ float running_max[kDecodeTile];
  __shared__ float running_sum[kDecodeTile];
  __shared__ float previous_scale[kDecodeTile];
  __shared__ float tile_quantized_sum[kDecodeTile];
  if (thread_idx < valid_rows) {
    running_max[thread_idx] = -FLT_MAX;
    running_sum[thread_idx] = 0.0f;
    previous_scale[thread_idx] = 1.0f;
    tile_quantized_sum[thread_idx] = 0.0f;
  }
  if (thread_idx == 0) {
    initialize_barrier(readable, 1);
  }
  __syncthreads();

  extern __shared__ std::uint8_t shared_memory[] alignas(1024);
  TensorStorage& storage = *reinterpret_cast<TensorStorage*>(shared_memory);
  auto sQuery =
      make_tensor(make_smem_ptr(storage.smem_X.begin()), SLayoutX{});
  auto sKey =
      make_tensor(make_smem_ptr(storage.smem_W.begin()), SLayoutW{});
  auto sQueryScale =
      make_tensor(make_smem_ptr(storage.smem_SFA.begin()), SLayoutSFA{});
  auto sKeyScale =
      make_tensor(make_smem_ptr(storage.smem_SFB.begin()), SLayoutSFB{});
  float* tile_logits = reinterpret_cast<float*>(shared_memory);

  auto layout_sfa =
      Config::make_layout_sfa(rows_per_group, max_kv_length,
                              head_dim, groups);
  auto layout_sfb =
      Config::make_layout_sfb(rows_per_group, max_kv_length,
                              head_dim, groups);
  auto value_layout_sfb =
      Config::make_layout_sfb(rows_per_group, value_dim,
                              max_kv_length, groups);
  auto mQuery = tma_query.get_tma_tensor(
      make_shape(rows_per_group, head_dim, groups));
  auto mKey = tma_key.get_tma_tensor(
      make_shape(max_kv_length, head_dim, groups));
  auto mValue = tma_value.get_tma_tensor(
      make_shape(value_dim, max_kv_length, groups));
  auto mQueryScale = tma_query_scale.get_tma_tensor(shape(layout_sfa));
  auto mKeyScale = tma_key_scale.get_tma_tensor(shape(layout_sfb));
  auto mValueScale =
      tma_value_scale.get_tma_tensor(shape(value_layout_sfb));

  using X = Underscore;
  using TileShape = typename Config::TileShape;
  auto gQuery_mkl = local_tile(
      mQuery, TileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});
  auto gKey_nkl = local_tile(
      mKey, TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});
  auto gQueryScale_mkl = local_tile(
      mQueryScale, TileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});
  auto gKeyScale_nkl = local_tile(
      mKeyScale, TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});
  auto gValue_nkl = local_tile(
      mValue, TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});
  auto gValueScale_nkl = local_tile(
      mValueScale, TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});

  auto query_tma_slice = tma_query.get_slice(0);
  auto key_tma_slice = tma_key.get_slice(0);
  auto query_scale_tma_slice = tma_query_scale.get_slice(0);
  auto key_scale_tma_slice = tma_key_scale.get_slice(0);
  auto value_tma_slice = tma_value.get_slice(0);
  auto value_scale_tma_slice = tma_value_scale.get_slice(0);
  auto tQuerysQuery = query_tma_slice.partition_D(sQuery);
  auto tKeysKey = key_tma_slice.partition_D(sKey);
  auto tQueryScalesQueryScale =
      query_scale_tma_slice.partition_D(sQueryScale);
  auto tKeyScalesKeyScale = key_scale_tma_slice.partition_D(sKeyScale);
  auto tValuesKey = value_tma_slice.partition_D(sKey);
  auto tValueScalesKeyScale =
      value_scale_tma_slice.partition_D(sKeyScale);

  auto gQuery = gQuery_mkl(_, _, row_tile, _, group);
  auto gQueryScale = gQueryScale_mkl(_, _, row_tile, _, group);
  auto tQuerygQuery = query_tma_slice.partition_S(gQuery);
  auto tQueryScalegQueryScale =
      query_scale_tma_slice.partition_S(gQueryScale);

  TiledMma tiled_mma;
  CollectiveMainloop collective_mainloop;
  auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
  auto tCrQuery = thread_mma.partition_fragment_A(sQuery(_, _, _0{}));
  auto tCrKey = thread_mma.partition_fragment_B(sKey(_, _, _0{}));
  auto tCrQueryScale = collective_mainloop.partition_fragment_SFA(
      sQueryScale(_, _, _0{}), thread_mma);
  auto tCrKeyScale = collective_mainloop.partition_fragment_SFB(
      sKeyScale(_, _, _0{}), thread_mma);

  auto query_copy =
      make_tiled_copy_A(typename Config::SmemCopyAtomX{}, tiled_mma);
  auto query_thread_copy = query_copy.get_thread_slice(thread_idx);
  auto tQuerysQueryCopy = query_thread_copy.partition_S(
      as_position_independent_swizzle_tensor(sQuery));
  auto tQueryrQuery = query_thread_copy.retile_D(tCrQuery);

  auto key_copy =
      make_tiled_copy_B(typename Config::SmemCopyAtomW{}, tiled_mma);
  auto key_thread_copy = key_copy.get_thread_slice(thread_idx);
  auto tKeysKeyCopy = key_thread_copy.partition_S(
      as_position_independent_swizzle_tensor(sKey));
  auto tKeyrKey = key_thread_copy.retile_D(tCrKey);

  auto tile_shape_mnk = tile_shape(tiled_mma);
  auto query_scale_copy = make_tiled_copy_impl(
      typename Config::SmemCopyAtomSFA{},
      collective_mainloop.get_layoutSFA_TV(tiled_mma),
      make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto query_scale_thread_copy =
      query_scale_copy.get_thread_slice(thread_idx);
  auto tQueryScalesQueryScaleCopy = query_scale_thread_copy.partition_S(
      as_position_independent_swizzle_tensor(sQueryScale));
  auto tQueryScalerQueryScale =
      query_scale_thread_copy.retile_D(tCrQueryScale);

  auto key_scale_copy = make_tiled_copy_impl(
      typename Config::SmemCopyAtomSFB{},
      collective_mainloop.get_layoutSFB_TV(tiled_mma),
      make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto key_scale_thread_copy =
      key_scale_copy.get_thread_slice(thread_idx);
  auto tKeyScalesKeyScaleCopy = key_scale_thread_copy.partition_S(
      as_position_independent_swizzle_tensor(sKeyScale));
  auto tKeyScalerKeyScale = key_scale_thread_copy.retile_D(tCrKeyScale);

  auto output_tile = make_tensor(
      make_gmem_ptr(static_cast<float*>(nullptr)),
      make_shape(Int<kTileM>{}, Int<kTileN>{}),
      make_stride(Int<kTileN>{}, _1{}));
  auto accum = thread_mma.partition_fragment_C(output_tile);
  auto identity = make_identity_tensor(shape(output_tile));
  auto output_coordinates = thread_mma.partition_C(identity);

  int barrier_phase = 0;
  int tile_count_k = ceil_div(head_dim, kTileK);
  int value_tiles = ceil_div(value_dim, kDecodeTile);

  for (int sequence_tile = tile_begin; sequence_tile < tile_end;
       ++sequence_tile) {
    clear(accum);
    auto gKey = gKey_nkl(_, _, sequence_tile, _, group);
    auto gKeyScale = gKeyScale_nkl(_, _, sequence_tile, _, group);
    auto tKeygKey = key_tma_slice.partition_S(gKey);
    auto tKeyScalegKeyScale = key_scale_tma_slice.partition_S(gKeyScale);

    for (int tile_k = 0; tile_k < tile_count_k; ++tile_k) {
      if constexpr (kPaged) {
        if (thread_idx == 0) {
          set_barrier_transaction_bytes(
              readable, CollectiveMainloop::TmaTransactionBytesMK);
          cute::copy(tma_query.with(readable),
                     tQuerygQuery(_, _, _, tile_k),
                     tQuerysQuery(_, _, _, _0{}));
          cute::copy(tma_query_scale.with(readable),
                     tQueryScalegQueryScale(_, _, _, tile_k),
                     tQueryScalesQueryScale(_, _, _, _0{}));
        }

        constexpr int kPackedColumns = kTileK / 2;
        constexpr int kTilePackedElements =
            kDecodeTile * kPackedColumns;
        for (int index = thread_idx; index < kTilePackedElements;
             index += blockDim.x) {
          int sequence_offset = index / kPackedColumns;
          int pair = index % kPackedColumns;
          int token = sequence_tile * kDecodeTile + sequence_offset;
          int dimension = tile_k * kTileK + pair * 2;
          std::uint8_t packed = 0;
          if (token < valid_length && dimension < head_dim) {
            int logical_block = token / block_size;
            int physical_block = block_table[
                static_cast<std::int64_t>(batch) *
                    max_blocks_per_sequence + logical_block];
            if (physical_block >= 0 && physical_block < physical_blocks) {
              int block_offset = token % block_size;
              std::int64_t packed_byte_offset =
                  (((static_cast<std::int64_t>(physical_block) * kv_heads +
                     kv_head) * block_size + block_offset) * head_dim +
                   dimension) / 2;
              packed = paged_key[packed_byte_offset];
            }
          }
          typename Config::Tin low;
          typename Config::Tin high;
          low.raw() = packed & 0xf;
          high.raw() = packed >> 4;
          sKey(sequence_offset, pair * 2, _0{}) = low;
          sKey(sequence_offset, pair * 2 + 1, _0{}) = high;
        }

        constexpr int kScaleBlocksPerTile =
            kTileK / kNvfp4ScaleVectorSize;
        constexpr int kTileScaleElements =
            kDecodeTile * kScaleBlocksPerTile;
        for (int index = thread_idx; index < kTileScaleElements;
             index += blockDim.x) {
          int sequence_offset = index / kScaleBlocksPerTile;
          int local_k_block = index % kScaleBlocksPerTile;
          int token = sequence_tile * kDecodeTile + sequence_offset;
          int dimension =
              (tile_k * kScaleBlocksPerTile + local_k_block) *
              kNvfp4ScaleVectorSize;
          typename Config::TS scale;
          scale.raw() = 0;
          if (token < valid_length && dimension < head_dim) {
            int logical_block = token / block_size;
            int physical_block = block_table[
                static_cast<std::int64_t>(batch) *
                    max_blocks_per_sequence + logical_block];
            if (physical_block >= 0 && physical_block < physical_blocks) {
              int block_offset = token % block_size;
              std::int64_t source_offset = fused_moe::sfb_offset(
                  block_offset,
                  dimension / kNvfp4ScaleVectorSize,
                  block_size, head_dim);
              scale.raw() = paged_key_scale[
                  (static_cast<std::int64_t>(physical_block) * kv_heads +
                   kv_head) * key_scale_stride + source_offset];
            }
          }
          sKeyScale(sequence_offset,
                    local_k_block * kNvfp4ScaleVectorSize, _0{}) = scale;
        }
        wait_barrier(readable, barrier_phase);
        __syncthreads();
      } else {
        if (thread_idx == 0) {
          set_barrier_transaction_bytes(
              readable, Config::kTmaTransactionBytes);
          cute::copy(tma_query.with(readable),
                     tQuerygQuery(_, _, _, tile_k),
                     tQuerysQuery(_, _, _, _0{}));
          cute::copy(tma_key.with(readable),
                     tKeygKey(_, _, _, tile_k),
                     tKeysKey(_, _, _, _0{}));
          cute::copy(tma_query_scale.with(readable),
                     tQueryScalegQueryScale(_, _, _, tile_k),
                     tQueryScalesQueryScale(_, _, _, _0{}));
          cute::copy(tma_key_scale.with(readable),
                     tKeyScalegKeyScale(_, _, _, tile_k),
                     tKeyScalesKeyScale(_, _, _, _0{}));
        }
        wait_barrier(readable, barrier_phase);
      }

      for_each(make_int_sequence<size<2>(tCrQuery)>{}, [&](auto k_block) {
        cute::copy(query_copy,
                   tQuerysQueryCopy(_, _, k_block, _0{}),
                   tQueryrQuery(_, _, k_block));
        cute::copy(key_copy,
                   tKeysKeyCopy(_, _, k_block, _0{}),
                   tKeyrKey(_, _, k_block));
        using MMAOp = typename TiledMma::MMA_Op;
        fp4_shift_A(MMAOp{}, tQueryrQuery(_, _, k_block));
        fp4_shift_B(MMAOp{}, tKeyrKey(_, _, k_block));
        cute::copy(tQueryScalesQueryScaleCopy(_, _, k_block, _0{}),
                   tQueryScalerQueryScale(_, _, k_block));
        cute::copy(tKeyScalesKeyScaleCopy(_, _, k_block, _0{}),
                   tKeyScalerKeyScale(_, _, k_block));
        cute::gemm(
            tiled_mma,
            make_zip_tensor(tCrQuery(_, _, k_block),
                            tCrQueryScale(_, _, k_block)),
            make_zip_tensor(tCrKey(_, _, k_block),
                            tCrKeyScale(_, _, k_block)),
            accum);
      });
      __syncthreads();
      barrier_phase ^= 1;
    }

#pragma unroll
    for (int i = 0; i < size(accum); ++i) {
      auto coordinate = output_coordinates(i);
      int row = get<0>(coordinate);
      int column = get<1>(coordinate);
      if (row < valid_rows) {
        tile_logits[row * kDecodeTile + column] = accum(i);
      }
    }
    __syncthreads();

    int tile_start = sequence_tile * kDecodeTile;
    int valid_columns = min(kDecodeTile, valid_length - tile_start);
    int warp = thread_idx / 32;
    int lane = thread_idx % 32;
    for (int row = warp; row < valid_rows; row += blockDim.x / 32) {
      float local_max = -FLT_MAX;
      for (int column = lane; column < valid_columns; column += 32) {
        local_max = fmaxf(
            local_max,
            tile_logits[row * kDecodeTile + column] * softmax_scale);
      }
      float tile_max = warp_reduce_max(local_max);
      tile_max = __shfl_sync(0xffffffff, tile_max, 0);
      float old_max = running_max[row];
      float new_max = fmaxf(old_max, tile_max);
      float alpha = isfinite(old_max) ? expf(old_max - new_max) : 0.0f;
      for (int column = lane; column < valid_columns; column += 32) {
        float probability = expf(
            tile_logits[row * kDecodeTile + column] * softmax_scale -
            new_max);
        tile_logits[row * kDecodeTile + column] = probability;
      }
      if (lane == 0) {
        previous_scale[row] = alpha;
        running_max[row] = new_max;
        tile_quantized_sum[row] = 0.0f;
      }
    }
    __syncthreads();

    constexpr int kProbabilityScaleBlocks =
        kDecodeTile / kNvfp4ScaleVectorSize;
    int probability_blocks = valid_rows * kProbabilityScaleBlocks;
    for (int index = thread_idx; index < probability_blocks;
         index += blockDim.x) {
      int row = index / kProbabilityScaleBlocks;
      int scale_block = index % kProbabilityScaleBlocks;
      int block_start = scale_block * kNvfp4ScaleVectorSize;
      float values[kNvfp4ScaleVectorSize];
      float amax = 0.0f;
#pragma unroll
      for (int element = 0; element < kNvfp4ScaleVectorSize; ++element) {
        int column = block_start + element;
        float probability = column < valid_columns
                                ? tile_logits[row * kDecodeTile + column]
                                : 0.0f;
        values[element] = probability;
        amax = fmaxf(amax, probability);
      }

      cutlass::float_ue4m3_t encoded_scale(0.0f);
      float decoded_scale = 1.0f;
      if (amax > 0.0f) {
        encoded_scale = cutlass::float_ue4m3_t(amax / 6.0f);
        decoded_scale = static_cast<float>(encoded_scale);
        if (!(decoded_scale > 0.0f)) {
          encoded_scale.raw() = 1;
          decoded_scale = static_cast<float>(encoded_scale);
        }
      }
      sQueryScale(row, block_start, _0{}) = encoded_scale;

      float reconstructed_sum = 0.0f;
#pragma unroll
      for (int pair_index = 0;
           pair_index < kNvfp4ScaleVectorSize / 2; ++pair_index) {
        float2 pair;
        pair.x = values[pair_index * 2] / decoded_scale;
        pair.y = values[pair_index * 2 + 1] / decoded_scale;
        __nv_fp4x2_e2m1 packed(pair);
        typename Config::Tin low;
        typename Config::Tin high;
        low.raw() = packed.__x & 0xf;
        high.raw() = packed.__x >> 4;
        sQuery(row, block_start + pair_index * 2, _0{}) = low;
        sQuery(row, block_start + pair_index * 2 + 1, _0{}) = high;
        float2 reconstructed = static_cast<float2>(packed);
        reconstructed_sum +=
            (reconstructed.x + reconstructed.y) * decoded_scale;
      }
      atomicAdd(tile_quantized_sum + row, reconstructed_sum);
    }
    __syncthreads();

    if (thread_idx < valid_rows) {
      running_sum[thread_idx] =
          running_sum[thread_idx] * previous_scale[thread_idx] +
          tile_quantized_sum[thread_idx];
    }
    __syncthreads();

    for (int value_tile = 0; value_tile < value_tiles; ++value_tile) {
      clear(accum);
      auto gValue = gValue_nkl(_, _, value_tile, _, group);
      auto gValueScale = gValueScale_nkl(_, _, value_tile, _, group);
      auto tValuegValue = value_tma_slice.partition_S(gValue);
      auto tValueScalegValueScale =
          value_scale_tma_slice.partition_S(gValueScale);
      if constexpr (kPaged) {
        constexpr int kPackedColumns = kTileK / 2;
        constexpr int kTilePackedElements =
            kDecodeTile * kPackedColumns;
        for (int index = thread_idx; index < kTilePackedElements;
             index += blockDim.x) {
          int value_offset = index / kPackedColumns;
          int pair = index % kPackedColumns;
          int value_column = value_tile * kDecodeTile + value_offset;
          int sequence_offset = pair * 2;
          int token = sequence_tile * kDecodeTile + sequence_offset;
          std::uint8_t packed = 0;
          if (value_column < value_dim && token < valid_length) {
            int logical_block = token / block_size;
            int physical_block = block_table[
                static_cast<std::int64_t>(batch) *
                    max_blocks_per_sequence + logical_block];
            if (physical_block >= 0 && physical_block < physical_blocks) {
              int block_offset = token % block_size;
              std::int64_t packed_byte_offset =
                  (((static_cast<std::int64_t>(physical_block) * kv_heads +
                     kv_head) * value_dim + value_column) * block_size +
                   block_offset) / 2;
              packed = paged_value[packed_byte_offset];
            }
          }
          typename Config::Tin low;
          typename Config::Tin high;
          low.raw() = packed & 0xf;
          high.raw() = packed >> 4;
          sKey(value_offset, sequence_offset, _0{}) = low;
          sKey(value_offset, sequence_offset + 1, _0{}) = high;
        }

        constexpr int kScaleBlocksPerTile =
            kTileK / kNvfp4ScaleVectorSize;
        constexpr int kTileScaleElements =
            kDecodeTile * kScaleBlocksPerTile;
        for (int index = thread_idx; index < kTileScaleElements;
             index += blockDim.x) {
          int value_offset = index / kScaleBlocksPerTile;
          int local_k_block = index % kScaleBlocksPerTile;
          int value_column = value_tile * kDecodeTile + value_offset;
          int token = sequence_tile * kDecodeTile +
                      local_k_block * kNvfp4ScaleVectorSize;
          typename Config::TS scale;
          scale.raw() = 0;
          if (value_column < value_dim && token < valid_length) {
            int logical_block = token / block_size;
            int physical_block = block_table[
                static_cast<std::int64_t>(batch) *
                    max_blocks_per_sequence + logical_block];
            if (physical_block >= 0 && physical_block < physical_blocks) {
              int block_offset = token % block_size;
              std::int64_t source_offset = fused_moe::sfb_offset(
                  value_column,
                  block_offset / kNvfp4ScaleVectorSize,
                  value_dim, block_size);
              scale.raw() = paged_value_scale[
                  (static_cast<std::int64_t>(physical_block) * kv_heads +
                   kv_head) * value_scale_stride + source_offset];
            }
          }
          sKeyScale(value_offset,
                    local_k_block * kNvfp4ScaleVectorSize, _0{}) = scale;
        }
        __syncthreads();
      } else {
        if (thread_idx == 0) {
          set_barrier_transaction_bytes(
              readable, CollectiveMainloop::TmaTransactionBytesNK);
          cute::copy(tma_value.with(readable),
                     tValuegValue(_, _, _, sequence_tile),
                     tValuesKey(_, _, _, _0{}));
          cute::copy(tma_value_scale.with(readable),
                     tValueScalegValueScale(_, _, _, sequence_tile),
                     tValueScalesKeyScale(_, _, _, _0{}));
        }
        wait_barrier(readable, barrier_phase);
      }

      for_each(make_int_sequence<size<2>(tCrQuery)>{}, [&](auto k_block) {
        cute::copy(query_copy,
                   tQuerysQueryCopy(_, _, k_block, _0{}),
                   tQueryrQuery(_, _, k_block));
        cute::copy(key_copy,
                   tKeysKeyCopy(_, _, k_block, _0{}),
                   tKeyrKey(_, _, k_block));
        using MMAOp = typename TiledMma::MMA_Op;
        fp4_shift_A(MMAOp{}, tQueryrQuery(_, _, k_block));
        fp4_shift_B(MMAOp{}, tKeyrKey(_, _, k_block));
        cute::copy(tQueryScalesQueryScaleCopy(_, _, k_block, _0{}),
                   tQueryScalerQueryScale(_, _, k_block));
        cute::copy(tKeyScalesKeyScaleCopy(_, _, k_block, _0{}),
                   tKeyScalerKeyScale(_, _, k_block));
        cute::gemm(
            tiled_mma,
            make_zip_tensor(tCrQuery(_, _, k_block),
                            tCrQueryScale(_, _, k_block)),
            make_zip_tensor(tCrKey(_, _, k_block),
                            tCrKeyScale(_, _, k_block)),
            accum);
      });
      __syncthreads();
      if constexpr (!kPaged) {
        barrier_phase ^= 1;
      }

#pragma unroll
      for (int i = 0; i < size(accum); ++i) {
        auto coordinate = output_coordinates(i);
        int row = get<0>(coordinate);
        int column = get<1>(coordinate);
        int value_column = value_tile * kDecodeTile + column;
        if (row < valid_rows && value_column < value_dim) {
          std::int64_t partial_index =
              ((first_matrix + row) * splits + split) * value_dim +
              value_column;
          partial_output[partial_index] =
              partial_output[partial_index] * previous_scale[row] +
              accum(i);
        }
      }
      __syncthreads();
    }
  }

  for (std::int64_t index = thread_idx; index < local_output_elements;
       index += blockDim.x) {
    int row = static_cast<int>(index / value_dim);
    int column = static_cast<int>(index % value_dim);
    std::int64_t matrix = first_matrix + row;
    std::int64_t partial_index =
        (matrix * splits + split) * value_dim + column;
    float normalized = running_sum[row] > 0.0f
                           ? partial_output[partial_index] / running_sum[row]
                           : 0.0f;
    if (splits == 1) {
      output[matrix * value_dim + column] = __float2half(normalized);
    } else {
      partial_output[partial_index] = normalized;
    }
  }
  if (splits > 1 && thread_idx < valid_rows) {
    std::int64_t matrix = first_matrix + thread_idx;
    partial_lse[matrix * splits + split] =
        running_sum[thread_idx] > 0.0f
            ? running_max[thread_idx] + logf(running_sum[thread_idx])
            : -FLT_MAX;
  }
}

__global__ void combine_decode_splits_kernel(
    const float* partial_output, const float* partial_lse,
    half* output, int matrices, int splits, int value_dim) {
  int matrix = blockIdx.x;
  if (matrix >= matrices) {
    return;
  }
  __shared__ float combined_max;
  __shared__ float combined_sum;
  if (threadIdx.x == 0) {
    float row_max = -FLT_MAX;
    for (int split = 0; split < splits; ++split) {
      row_max = fmaxf(row_max, partial_lse[matrix * splits + split]);
    }
    float row_sum = 0.0f;
    if (isfinite(row_max)) {
      for (int split = 0; split < splits; ++split) {
        row_sum += expf(partial_lse[matrix * splits + split] - row_max);
      }
    }
    combined_max = row_max;
    combined_sum = row_sum;
  }
  __syncthreads();

  for (int column = threadIdx.x; column < value_dim;
       column += blockDim.x) {
    float result = 0.0f;
    if (combined_sum > 0.0f) {
      for (int split = 0; split < splits; ++split) {
        float weight = expf(
            partial_lse[matrix * splits + split] - combined_max);
        result += weight *
            partial_output[(static_cast<std::int64_t>(matrix) * splits +
                            split) * value_dim + column];
      }
      result /= combined_sum;
    }
    output[static_cast<std::int64_t>(matrix) * value_dim + column] =
        __float2half(result);
  }
}

bool valid_problem(
    int batch, int heads, int query_length, int kv_length,
    int head_dim, int value_dim, bool causal, float softmax_scale) {
  std::int64_t matrices =
      static_cast<std::int64_t>(batch) * static_cast<std::int64_t>(heads);
  return batch > 0 && heads > 0 && matrices <= std::numeric_limits<int>::max() &&
         query_length > 0 && kv_length > 0 &&
         head_dim > 0 && value_dim > 0 &&
         head_dim % kInputAlignmentElements == 0 &&
         kv_length % kInputAlignmentElements == 0 &&
         value_dim % kOutputAlignmentElements == 0 &&
         (!causal || query_length <= kv_length) &&
         std::isfinite(softmax_scale) && softmax_scale > 0.0f;
}

}  // namespace attention_detail

std::size_t nvfp4_attention_workspace_size_sm120(
    int batch, int heads, int query_length, int kv_length) {
  if (batch <= 0 || heads <= 0 || query_length <= 0 || kv_length <= 0 ||
      kv_length % kInputAlignmentElements != 0) {
    return 0;
  }
  std::size_t matrices =
      static_cast<std::size_t>(batch) * static_cast<std::size_t>(heads);
  std::size_t matrix_elements =
      static_cast<std::size_t>(query_length) *
      static_cast<std::size_t>(kv_length);
  std::size_t offset = 0;
  offset = attention_detail::align_workspace(offset);
  offset += matrices * matrix_elements * sizeof(float);
  offset = attention_detail::align_workspace(offset);
  offset += matrices * packed_fp4_bytes(query_length, kv_length);
  offset = attention_detail::align_workspace(offset);
  offset += matrices * scale_a_elements(query_length, 1, kv_length);
  offset = attention_detail::align_workspace(offset);
  offset += matrices * static_cast<std::size_t>(query_length) * sizeof(float);
  return attention_detail::align_workspace(offset);
}

GemmStatus nvfp4_attention_prefill_sm120(
    int batch, int heads, int query_length, int kv_length,
    int head_dim, int value_dim,
    const void* query, const void* key, const void* value_transposed,
    const void* query_scale, const void* key_scale,
    const void* value_scale,
    bool causal, float softmax_scale,
    half* output,
    void* workspace, std::size_t workspace_bytes,
    cudaStream_t stream) {
  if (!attention_detail::valid_problem(
          batch, heads, query_length, kv_length, head_dim, value_dim,
          causal, softmax_scale) ||
      query == nullptr || key == nullptr || value_transposed == nullptr ||
      query_scale == nullptr || key_scale == nullptr ||
      value_scale == nullptr || output == nullptr || workspace == nullptr) {
    return GemmStatus::kInvalidArgument;
  }

  std::size_t required = nvfp4_attention_workspace_size_sm120(
      batch, heads, query_length, kv_length);
  if (required == 0 || workspace_bytes < required) {
    return GemmStatus::kInsufficientWorkspace;
  }

  std::size_t matrices =
      static_cast<std::size_t>(batch) * static_cast<std::size_t>(heads);
  std::size_t logits_elements_per_matrix =
      static_cast<std::size_t>(query_length) * kv_length;
  std::size_t logits_bytes =
      matrices * logits_elements_per_matrix * sizeof(float);
  std::size_t probability_bytes_per_matrix =
      packed_fp4_bytes(query_length, kv_length);
  std::size_t probability_scale_elements_per_matrix =
      scale_a_elements(query_length, value_dim, kv_length);

  auto* workspace_bytes_ptr = static_cast<std::uint8_t*>(workspace);
  std::size_t offset = 0;
  offset = attention_detail::align_workspace(offset);
  float* logits = reinterpret_cast<float*>(workspace_bytes_ptr + offset);
  offset += logits_bytes;
  offset = attention_detail::align_workspace(offset);
  std::uint8_t* probabilities = workspace_bytes_ptr + offset;
  offset += matrices * probability_bytes_per_matrix;
  offset = attention_detail::align_workspace(offset);
  auto* probability_scale =
      reinterpret_cast<cutlass::float_ue4m3_t*>(workspace_bytes_ptr + offset);
  offset += matrices * probability_scale_elements_per_matrix;
  offset = attention_detail::align_workspace(offset);
  float* row_correction =
      reinterpret_cast<float*>(workspace_bytes_ptr + offset);

  const auto* query_bytes = static_cast<const std::uint8_t*>(query);
  const auto* key_bytes = static_cast<const std::uint8_t*>(key);
  const auto* value_bytes =
      static_cast<const std::uint8_t*>(value_transposed);
  const auto* query_scale_bytes =
      static_cast<const std::uint8_t*>(query_scale);
  const auto* key_scale_bytes = static_cast<const std::uint8_t*>(key_scale);
  const auto* value_scale_bytes =
      static_cast<const std::uint8_t*>(value_scale);

  std::size_t query_bytes_per_matrix =
      packed_fp4_bytes(query_length, head_dim);
  std::size_t key_bytes_per_matrix = packed_fp4_bytes(kv_length, head_dim);
  std::size_t value_bytes_per_matrix =
      packed_fp4_bytes(value_dim, kv_length);
  std::size_t query_scale_elements_per_matrix =
      scale_a_elements(query_length, kv_length, head_dim);
  std::size_t key_scale_elements_per_matrix =
      scale_b_elements(query_length, kv_length, head_dim);
  std::size_t value_scale_elements_per_matrix =
      scale_b_elements(query_length, value_dim, kv_length);

  for (std::size_t matrix = 0; matrix < matrices; ++matrix) {
    GemmStatus status = nvfp4_cute_gemm_f32_sm120(
        query_length, kv_length, head_dim,
        query_bytes + matrix * query_bytes_per_matrix,
        key_bytes + matrix * key_bytes_per_matrix,
        query_scale_bytes + matrix * query_scale_elements_per_matrix,
        key_scale_bytes + matrix * key_scale_elements_per_matrix,
        logits + matrix * logits_elements_per_matrix, stream);
    if (status != GemmStatus::kSuccess) {
      return status;
    }
  }

  cudaError_t error = cudaMemsetAsync(
      probability_scale, 0,
      matrices * probability_scale_elements_per_matrix, stream);
  if (error != cudaSuccess) {
    return GemmStatus::kCudaError;
  }

  std::int64_t total_rows =
      static_cast<std::int64_t>(matrices) * query_length;
  int blocks = static_cast<int>(std::min<std::int64_t>(total_rows, 65535));
  attention_detail::softmax_quantize_nvfp4_kernel<<<
      blocks, attention_detail::kSoftmaxThreads, 0, stream>>>(
      logits, probabilities, probability_scale, row_correction,
      static_cast<int>(matrices), query_length, kv_length,
      heads, /*kv_lengths=*/nullptr,
      align_up(query_length, kScaleMNAlignment),
      static_cast<std::int64_t>(probability_scale_elements_per_matrix),
      causal, softmax_scale);
  if (cudaPeekAtLastError() != cudaSuccess) {
    return GemmStatus::kCudaError;
  }

  std::size_t output_elements_per_matrix =
      static_cast<std::size_t>(query_length) * value_dim;
  for (std::size_t matrix = 0; matrix < matrices; ++matrix) {
    GemmStatus status = nvfp4_cute_gemm_sm120(
        query_length, value_dim, kv_length,
        probabilities + matrix * probability_bytes_per_matrix,
        value_bytes + matrix * value_bytes_per_matrix,
        probability_scale + matrix * probability_scale_elements_per_matrix,
        value_scale_bytes + matrix * value_scale_elements_per_matrix,
        output + matrix * output_elements_per_matrix, stream);
    if (status != GemmStatus::kSuccess) {
      return status;
    }
  }

  std::int64_t output_elements =
      static_cast<std::int64_t>(matrices) * query_length * value_dim;
  int correction_blocks = static_cast<int>(std::min<std::int64_t>(
      (output_elements + attention_detail::kSoftmaxThreads - 1) /
          attention_detail::kSoftmaxThreads,
      65535));
  attention_detail::apply_row_correction_kernel<<<
      correction_blocks, attention_detail::kSoftmaxThreads, 0, stream>>>(
      output, row_correction, output_elements, value_dim);
  if (cudaPeekAtLastError() != cudaSuccess) {
    return GemmStatus::kCudaError;
  }
  return GemmStatus::kSuccess;
}

std::size_t nvfp4_attention_decode_workspace_size_sm120(
    int batch, int query_heads, int kv_heads, int max_kv_length,
    int head_dim, int value_dim) {
  if (batch <= 0 || query_heads <= 0 || kv_heads <= 0 ||
      query_heads % kv_heads != 0 || max_kv_length <= 0 ||
      max_kv_length % kInputAlignmentElements != 0 || head_dim <= 0 ||
      head_dim % kInputAlignmentElements != 0 || value_dim <= 0 ||
      value_dim % kOutputAlignmentElements != 0) {
    return 0;
  }
  std::size_t matrices =
      static_cast<std::size_t>(batch) * query_heads;
  std::size_t groups = static_cast<std::size_t>(batch) * kv_heads;
  int rows_per_group = query_heads / kv_heads;
  int splits = attention_detail::decode_split_count(
      batch, kv_heads, rows_per_group, max_kv_length);
  std::size_t offset = 0;
  offset = attention_detail::align_workspace(offset);
  offset += groups * scale_a_elements(
      rows_per_group, max_kv_length, head_dim);
  offset = attention_detail::align_workspace(offset);
  offset += matrices * static_cast<std::size_t>(splits) * value_dim *
            sizeof(float);
  if (splits > 1) {
    offset = attention_detail::align_workspace(offset);
    offset += matrices * static_cast<std::size_t>(splits) * sizeof(float);
  }
  return attention_detail::align_workspace(offset);
}

GemmStatus nvfp4_attention_decode_sm120(
    int batch, int query_heads, int kv_heads, int max_kv_length,
    int head_dim, int value_dim,
    const void* query, const void* key_cache,
    const void* value_cache_transposed,
    const void* query_scale, const void* key_scale,
    const void* value_scale,
    const int* kv_lengths, float softmax_scale,
    half* output,
    void* workspace, std::size_t workspace_bytes,
    cudaStream_t stream) {
  std::int64_t kv_matrices =
      static_cast<std::int64_t>(batch) * static_cast<std::int64_t>(kv_heads);
  if (!attention_detail::valid_problem(
          batch, query_heads, /*query_length=*/1, max_kv_length,
          head_dim, value_dim, /*causal=*/false, softmax_scale) ||
      kv_heads <= 0 || query_heads % kv_heads != 0 ||
      kv_matrices > std::numeric_limits<int>::max() ||
      query == nullptr || key_cache == nullptr ||
      value_cache_transposed == nullptr || query_scale == nullptr ||
      key_scale == nullptr || value_scale == nullptr || output == nullptr ||
      workspace == nullptr) {
    return GemmStatus::kInvalidArgument;
  }

  std::size_t required = nvfp4_attention_decode_workspace_size_sm120(
      batch, query_heads, kv_heads, max_kv_length, head_dim, value_dim);
  if (required == 0 || workspace_bytes < required) {
    return GemmStatus::kInsufficientWorkspace;
  }

  const int heads_per_kv = query_heads / kv_heads;
  const std::size_t matrices =
      static_cast<std::size_t>(batch) * query_heads;
  const std::size_t groups =
      static_cast<std::size_t>(batch) * kv_heads;
  const std::size_t grouped_query_scale_elements =
      scale_a_elements(heads_per_kv, max_kv_length, head_dim);
  const int splits = attention_detail::decode_split_count(
      batch, kv_heads, heads_per_kv, max_kv_length);

  auto* workspace_bytes_ptr = static_cast<std::uint8_t*>(workspace);
  std::size_t offset = 0;
  offset = attention_detail::align_workspace(offset);
  auto* grouped_query_scale = workspace_bytes_ptr + offset;
  offset += groups * grouped_query_scale_elements;
  offset = attention_detail::align_workspace(offset);
  float* partial_output =
      reinterpret_cast<float*>(workspace_bytes_ptr + offset);
  offset += matrices * static_cast<std::size_t>(splits) * value_dim *
            sizeof(float);
  float* partial_lse = nullptr;
  if (splits > 1) {
    offset = attention_detail::align_workspace(offset);
    partial_lse = reinterpret_cast<float*>(workspace_bytes_ptr + offset);
  }

  const auto* query_bytes = static_cast<const std::uint8_t*>(query);
  const auto* key_bytes = static_cast<const std::uint8_t*>(key_cache);
  const auto* value_bytes =
      static_cast<const std::uint8_t*>(value_cache_transposed);
  const auto* query_scale_bytes =
      static_cast<const std::uint8_t*>(query_scale);
  const auto* key_scale_bytes = static_cast<const std::uint8_t*>(key_scale);
  const auto* value_scale_bytes =
      static_cast<const std::uint8_t*>(value_scale);

  const std::size_t query_scale_elements_per_matrix =
      scale_a_elements(/*m=*/1, max_kv_length, head_dim);

  std::int64_t query_scale_values =
      static_cast<std::int64_t>(groups) * heads_per_kv *
      (head_dim / kNvfp4ScaleVectorSize);
  int repack_blocks = static_cast<int>(std::min<std::int64_t>(
      (query_scale_values + attention_detail::kSoftmaxThreads - 1) /
          attention_detail::kSoftmaxThreads,
      65535));
  attention_detail::repack_decode_query_scales_kernel<<<
      repack_blocks, attention_detail::kSoftmaxThreads, 0, stream>>>(
      query_scale_bytes, grouped_query_scale, static_cast<int>(groups),
      heads_per_kv, head_dim,
      static_cast<std::int64_t>(query_scale_elements_per_matrix),
      static_cast<std::int64_t>(grouped_query_scale_elements));
  if (cudaPeekAtLastError() != cudaSuccess) {
    return GemmStatus::kCudaError;
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

  using Config =
      detail::Nvfp4GemmConfig<float, attention_detail::kDecodeTile,
                              attention_detail::kDecodeTile, 128, 2>;
  using TmaInternalElementX = typename Config::TmaInternalElementX;
  using TmaInternalElementW = typename Config::TmaInternalElementW;
  using Scale = typename Config::TS;
  Config config;
  auto query_tensor = cute::make_tensor(
      cute::recast_ptr<TmaInternalElementX>(query_bytes),
      cute::make_shape(int32_t(heads_per_kv), int32_t(head_dim),
                       int32_t(groups)),
      typename Config::StrideX{int64_t(head_dim), cute::Int<1>{},
                               int64_t(heads_per_kv) * head_dim});
  auto key_tensor = cute::make_tensor(
      cute::recast_ptr<TmaInternalElementW>(key_bytes),
      cute::make_shape(int32_t(max_kv_length), int32_t(head_dim),
                       int32_t(groups)),
      typename Config::StrideW{int64_t(head_dim), cute::Int<1>{},
                               int64_t(max_kv_length) * head_dim});
  auto value_tensor = cute::make_tensor(
      cute::recast_ptr<TmaInternalElementW>(value_bytes),
      cute::make_shape(int32_t(value_dim), int32_t(max_kv_length),
                       int32_t(groups)),
      typename Config::StrideW{int64_t(max_kv_length), cute::Int<1>{},
                               int64_t(value_dim) * max_kv_length});
  auto query_scale_layout = Config::make_layout_sfa(
      heads_per_kv, max_kv_length, head_dim, static_cast<int>(groups));
  auto key_scale_layout = Config::make_layout_sfb(
      heads_per_kv, max_kv_length, head_dim, static_cast<int>(groups));
  auto value_scale_layout = Config::make_layout_sfb(
      heads_per_kv, value_dim, max_kv_length,
      static_cast<int>(groups));
  auto query_scale_tensor = cute::make_tensor(
      reinterpret_cast<const Scale*>(grouped_query_scale),
      query_scale_layout);
  auto key_scale_tensor = cute::make_tensor(
      reinterpret_cast<const Scale*>(key_scale_bytes), key_scale_layout);
  auto value_scale_tensor = cute::make_tensor(
      reinterpret_cast<const Scale*>(value_scale_bytes), value_scale_layout);
  auto tma = config.get_tma(
      query_tensor, key_tensor, query_scale_tensor, key_scale_tensor);
  auto value_tma = config.get_tma(
      query_tensor, value_tensor, query_scale_tensor, value_scale_tensor);

  int row_tiles = ceil_div(heads_per_kv, attention_detail::kDecodeTile);
  std::int64_t tasks = static_cast<std::int64_t>(groups) * row_tiles;
  if (tasks > std::numeric_limits<int>::max()) {
    return GemmStatus::kInvalidArgument;
  }
  constexpr int kSharedMemoryBytes =
      attention_detail::kDecodeTile * attention_detail::kDecodeTile *
      sizeof(float);
  auto kernel =
      attention_detail::fused_decode_nvfp4_kernel<Config, false>;
  if (cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
          kSharedMemoryBytes) != cudaSuccess) {
    return GemmStatus::kCudaError;
  }
  dim3 grid(static_cast<unsigned int>(tasks),
            static_cast<unsigned int>(splits));
  kernel<<<grid, attention_detail::kSoftmaxThreads,
           kSharedMemoryBytes, stream>>>(
      tma.x, tma.w, tma.sfa, tma.sfb, value_tma.w, value_tma.sfb,
      /*paged_key=*/nullptr, /*paged_value=*/nullptr,
      /*paged_key_scale=*/nullptr, /*paged_value_scale=*/nullptr,
      /*block_table=*/nullptr,
      kv_lengths, softmax_scale, partial_output, partial_lse, output,
      static_cast<int>(groups), kv_heads, heads_per_kv,
      max_kv_length, head_dim, value_dim, splits,
      /*physical_blocks=*/0, /*max_blocks_per_sequence=*/0,
      /*block_size=*/0, /*key_scale_stride=*/0,
      /*value_scale_stride=*/0);
  if (cudaPeekAtLastError() != cudaSuccess) {
    return GemmStatus::kCudaError;
  }
  if (splits > 1) {
    attention_detail::combine_decode_splits_kernel<<<
        static_cast<int>(matrices), attention_detail::kSoftmaxThreads,
        0, stream>>>(
        partial_output, partial_lse, output, static_cast<int>(matrices),
        splits, value_dim);
    if (cudaPeekAtLastError() != cudaSuccess) {
      return GemmStatus::kCudaError;
    }
  }
  return GemmStatus::kSuccess;
}

GemmStatus nvfp4_attention_paged_decode_sm120(
    int batch, int query_heads, int kv_heads, int physical_blocks,
    int max_blocks_per_sequence, int block_size,
    int head_dim, int value_dim,
    const void* query, const void* key_cache,
    const void* value_cache_transposed,
    const void* query_scale, const void* key_scale,
    const void* value_scale,
    const int* block_table, const int* kv_lengths, float softmax_scale,
    half* output,
    void* workspace, std::size_t workspace_bytes,
    cudaStream_t stream) {
  if (max_blocks_per_sequence <= 0 || block_size <= 0 ||
      max_blocks_per_sequence >
          std::numeric_limits<int>::max() / block_size) {
    return GemmStatus::kInvalidArgument;
  }
  int max_kv_length = max_blocks_per_sequence * block_size;
  std::int64_t kv_matrices =
      static_cast<std::int64_t>(batch) * kv_heads;
  std::int64_t page_groups =
      static_cast<std::int64_t>(physical_blocks) * kv_heads;
  if (!attention_detail::valid_problem(
          batch, query_heads, /*query_length=*/1, max_kv_length,
          head_dim, value_dim, /*causal=*/false, softmax_scale) ||
      kv_heads <= 0 || query_heads % kv_heads != 0 ||
      physical_blocks <= 0 ||
      (block_size != 32 && block_size != 64 && block_size != 128) ||
      kv_matrices > std::numeric_limits<int>::max() ||
      page_groups > std::numeric_limits<int>::max() ||
      query == nullptr || key_cache == nullptr ||
      value_cache_transposed == nullptr || query_scale == nullptr ||
      key_scale == nullptr || value_scale == nullptr ||
      block_table == nullptr || kv_lengths == nullptr || output == nullptr ||
      workspace == nullptr) {
    return GemmStatus::kInvalidArgument;
  }

  std::size_t required = nvfp4_attention_decode_workspace_size_sm120(
      batch, query_heads, kv_heads, max_kv_length, head_dim, value_dim);
  if (required == 0 || workspace_bytes < required) {
    return GemmStatus::kInsufficientWorkspace;
  }

  const int heads_per_kv = query_heads / kv_heads;
  const std::size_t matrices =
      static_cast<std::size_t>(batch) * query_heads;
  const std::size_t groups =
      static_cast<std::size_t>(batch) * kv_heads;
  const std::size_t grouped_query_scale_elements =
      scale_a_elements(heads_per_kv, max_kv_length, head_dim);
  const std::size_t key_scale_elements_per_page =
      scale_b_elements(/*m=*/1, block_size, head_dim);
  const std::size_t value_scale_elements_per_page =
      scale_b_elements(/*m=*/1, value_dim, block_size);
  if (key_scale_elements_per_page >
          static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
      value_scale_elements_per_page >
          static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    return GemmStatus::kInvalidArgument;
  }
  const int splits = attention_detail::decode_split_count(
      batch, kv_heads, heads_per_kv, max_kv_length);

  auto* workspace_bytes_ptr = static_cast<std::uint8_t*>(workspace);
  std::size_t offset = 0;
  offset = attention_detail::align_workspace(offset);
  auto* grouped_query_scale = workspace_bytes_ptr + offset;
  offset += groups * grouped_query_scale_elements;
  offset = attention_detail::align_workspace(offset);
  float* partial_output =
      reinterpret_cast<float*>(workspace_bytes_ptr + offset);
  offset += matrices * static_cast<std::size_t>(splits) * value_dim *
            sizeof(float);
  float* partial_lse = nullptr;
  if (splits > 1) {
    offset = attention_detail::align_workspace(offset);
    partial_lse = reinterpret_cast<float*>(workspace_bytes_ptr + offset);
  }

  const auto* query_bytes = static_cast<const std::uint8_t*>(query);
  const auto* key_bytes = static_cast<const std::uint8_t*>(key_cache);
  const auto* value_bytes =
      static_cast<const std::uint8_t*>(value_cache_transposed);
  const auto* query_scale_bytes =
      static_cast<const std::uint8_t*>(query_scale);
  const auto* key_scale_bytes = static_cast<const std::uint8_t*>(key_scale);
  const auto* value_scale_bytes =
      static_cast<const std::uint8_t*>(value_scale);

  const std::size_t query_scale_elements_per_matrix =
      scale_a_elements(/*m=*/1, max_kv_length, head_dim);
  std::int64_t query_scale_values =
      static_cast<std::int64_t>(groups) * heads_per_kv *
      (head_dim / kNvfp4ScaleVectorSize);
  int repack_blocks = static_cast<int>(std::min<std::int64_t>(
      (query_scale_values + attention_detail::kSoftmaxThreads - 1) /
          attention_detail::kSoftmaxThreads,
      65535));
  attention_detail::repack_decode_query_scales_kernel<<<
      repack_blocks, attention_detail::kSoftmaxThreads, 0, stream>>>(
      query_scale_bytes, grouped_query_scale, static_cast<int>(groups),
      heads_per_kv, head_dim,
      static_cast<std::int64_t>(query_scale_elements_per_matrix),
      static_cast<std::int64_t>(grouped_query_scale_elements));
  if (cudaPeekAtLastError() != cudaSuccess) {
    return GemmStatus::kCudaError;
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

  using Config =
      detail::Nvfp4GemmConfig<float, attention_detail::kDecodeTile,
                              attention_detail::kDecodeTile, 128, 2>;
  using TmaInternalElementX = typename Config::TmaInternalElementX;
  using TmaInternalElementW = typename Config::TmaInternalElementW;
  using Scale = typename Config::TS;
  Config config;
  auto query_tensor = cute::make_tensor(
      cute::recast_ptr<TmaInternalElementX>(query_bytes),
      cute::make_shape(int32_t(heads_per_kv), int32_t(head_dim),
                       int32_t(groups)),
      typename Config::StrideX{int64_t(head_dim), cute::Int<1>{},
                               int64_t(heads_per_kv) * head_dim});
  auto key_tensor = cute::make_tensor(
      cute::recast_ptr<TmaInternalElementW>(key_bytes),
      cute::make_shape(int32_t(block_size), int32_t(head_dim),
                       int32_t(page_groups)),
      typename Config::StrideW{int64_t(head_dim), cute::Int<1>{},
                               int64_t(block_size) * head_dim});
  auto value_tensor = cute::make_tensor(
      cute::recast_ptr<TmaInternalElementW>(value_bytes),
      cute::make_shape(int32_t(value_dim), int32_t(block_size),
                       int32_t(page_groups)),
      typename Config::StrideW{int64_t(block_size), cute::Int<1>{},
                               int64_t(value_dim) * block_size});
  auto query_scale_layout = Config::make_layout_sfa(
      heads_per_kv, max_kv_length, head_dim, static_cast<int>(groups));
  auto key_scale_layout = Config::make_layout_sfb(
      heads_per_kv, block_size, head_dim, static_cast<int>(page_groups));
  auto value_scale_layout = Config::make_layout_sfb(
      heads_per_kv, value_dim, block_size,
      static_cast<int>(page_groups));
  auto query_scale_tensor = cute::make_tensor(
      reinterpret_cast<const Scale*>(grouped_query_scale),
      query_scale_layout);
  auto key_scale_tensor = cute::make_tensor(
      reinterpret_cast<const Scale*>(key_scale_bytes), key_scale_layout);
  auto value_scale_tensor = cute::make_tensor(
      reinterpret_cast<const Scale*>(value_scale_bytes), value_scale_layout);
  auto tma = config.get_tma(
      query_tensor, key_tensor, query_scale_tensor, key_scale_tensor);
  auto value_tma = config.get_tma(
      query_tensor, value_tensor, query_scale_tensor, value_scale_tensor);

  int row_tiles = ceil_div(heads_per_kv, attention_detail::kDecodeTile);
  std::int64_t tasks = static_cast<std::int64_t>(groups) * row_tiles;
  if (tasks > std::numeric_limits<int>::max()) {
    return GemmStatus::kInvalidArgument;
  }
  constexpr int kSharedMemoryBytes =
      attention_detail::kDecodeTile * attention_detail::kDecodeTile *
      sizeof(float);
  dim3 grid(static_cast<unsigned int>(tasks),
            static_cast<unsigned int>(splits));
  auto kernel =
      attention_detail::fused_decode_nvfp4_kernel<Config, true>;
  if (cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
          kSharedMemoryBytes) != cudaSuccess) {
    return GemmStatus::kCudaError;
  }
  kernel<<<grid, attention_detail::kSoftmaxThreads,
           kSharedMemoryBytes, stream>>>(
      tma.x, tma.w, tma.sfa, tma.sfb, value_tma.w, value_tma.sfb,
      key_bytes, value_bytes, key_scale_bytes, value_scale_bytes,
      block_table, kv_lengths, softmax_scale,
      partial_output, partial_lse, output,
      static_cast<int>(groups), kv_heads, heads_per_kv,
      max_kv_length, head_dim, value_dim, splits,
      physical_blocks, max_blocks_per_sequence, block_size,
      static_cast<std::int64_t>(key_scale_elements_per_page),
      static_cast<std::int64_t>(value_scale_elements_per_page));
  if (cudaPeekAtLastError() != cudaSuccess) {
    return GemmStatus::kCudaError;
  }
  if (splits > 1) {
    attention_detail::combine_decode_splits_kernel<<<
        static_cast<int>(matrices), attention_detail::kSoftmaxThreads,
        0, stream>>>(
        partial_output, partial_lse, output, static_cast<int>(matrices),
        splits, value_dim);
    if (cudaPeekAtLastError() != cudaSuccess) {
      return GemmStatus::kCudaError;
    }
  }
  return GemmStatus::kSuccess;
}

}  // namespace sm120_nvfp4
