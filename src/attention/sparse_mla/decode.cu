#include "sm120_nvfp4/sparse_mla.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include "mma.cuh"

namespace sm120_nvfp4::sparse_mla {

constexpr int kHeads = 64, kDim = 512, kNope = 448, kRope = 64;
constexpr int kCandidates = 64, kWarps = 8, kThreads = 256;
constexpr float kLog2e = 1.4426950408889634f;
constexpr float kEmpty = -1.0e30f;

struct alignas(16) RawTile {
  std::uint8_t data[64][352];
  std::uint8_t scale[64][32];
  int valid[64];
};

struct alignas(16) SharedStorage {
  std::uint8_t q[64][224];
  std::uint8_t qs[64][28];
  RawTile raw[2];
  std::uint8_t vt[448][32];
  std::uint8_t vts[448][4];
  __nv_bfloat16 weight[64][64];
  std::uint8_t p[64][32];
  std::uint8_t ps[64][4];
  float local_max[64][8];
  float local_sum[64][8];
  float maximum[64], denominator[64], alpha[64];
};
static_assert(sizeof(SharedStorage) <= 99 * 1024, "SM120 shared memory limit");

__device__ __forceinline__ float scale_value(std::uint8_t code) {
  __nv_fp8_e4m3 s;
  s.__x = code;
  return static_cast<float>(s);
}

__device__ __forceinline__ void quantize16(const float (&x)[16],
                                          std::uint8_t* dst,
                                          std::uint8_t* scale_dst) {
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(x[i]));
  __nv_fp8_e4m3 scale(amax / 6.f);
  *scale_dst = scale.__x;
  float sf = static_cast<float>(scale);
  float inv = sf == 0.f ? 0.f : 1.f / sf;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    __nv_fp4x2_e2m1 pair(make_float2(x[2 * i] * inv, x[2 * i + 1] * inv));
    dst[i] = pair.__x;
  }
}

// Slot IDs already encode physical page and row; each pool has its own IDs.
__device__ __forceinline__ int slot_for(const SparseMlaDecodeParams& p,
                                       int batch, int chunk, int row,
                                       bool& swa) {
  int swa_chunks = (p.swa_candidates + 63) / 64;
  swa = chunk < swa_chunks;
  int capacity = swa ? p.swa_candidates : p.compressed_candidates;
  int position = (swa ? chunk : chunk - swa_chunks) * 64 + row;
  const int* lengths = swa ? p.swa_lengths : p.compressed_lengths;
  int length = lengths ? max(0, min(capacity, lengths[batch])) : capacity;
  if (position >= length) return -1;
  const int* indices = swa ? p.swa_indices : p.compressed_indices;
  int slot = indices[static_cast<std::int64_t>(batch) * capacity + position];
  int pages = swa ? p.swa_pages : p.compressed_pages;
  return slot >= 0 && static_cast<std::int64_t>(slot) < static_cast<std::int64_t>(pages) * 64
             ? slot : -1;
}

// All 256 threads commit one async-copy group, including threads with only
// masked rows. wait_prior(0)+CTA barrier is mandatory before consuming it.
__device__ __forceinline__ void prefetch(const SparseMlaDecodeParams& p, RawTile& tile,
                         int batch, int chunk) {
  for (int v = threadIdx.x; v < 64 * 24; v += kThreads) {
    int row = v / 24, vec = v % 24;
    bool swa;
    int slot = slot_for(p, batch, chunk, row, swa);
    const std::uint8_t* source = reinterpret_cast<const std::uint8_t*>(p.query);
    if (slot >= 0) {
      const std::uint8_t* pool = swa ? p.swa_cache : p.compressed_cache;
      source = pool + static_cast<std::int64_t>(slot / 64) * (64 * 384);
      source += vec < 22 ? (slot % 64) * 352 + vec * 16
                        : 64 * 352 + (slot % 64) * 32 + (vec - 22) * 16;
    }
    std::uint8_t* dest = vec < 22 ? tile.data[row] + vec * 16
                                : tile.scale[row] + (vec - 22) * 16;
    __pipeline_memcpy_async(dest, source, 16, slot < 0 ? 16 : 0);
  }
  if (threadIdx.x < 64) {
    bool swa;
    tile.valid[threadIdx.x] = slot_for(p, batch, chunk, threadIdx.x, swa) >= 0;
  }
  __pipeline_commit();
}

__device__ __forceinline__ void prepare_v(const RawTile& raw, SharedStorage& sm) {
  // One worker owns an entire group of 16 candidates for one output channel.
  // Cache scales are absorbed BEFORE requantizing along the candidate axis.
  for (int i = threadIdx.x; i < 448 * 4; i += kThreads) {
    int dim = i / 4, group = i % 4;
    float values[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      int row = group * 16 + j;
      __nv_fp4x2_e2m1 packed;
      packed.__x = raw.data[row][dim / 2];
      float2 decoded = static_cast<float2>(packed);
      values[j] = ((dim & 1) ? decoded.y : decoded.x) *
                  scale_value(raw.scale[row][dim / 16]);
    }
    quantize16(values, sm.vt[dim] + group * 8, sm.vts[dim] + group);
  }
}

__device__ __forceinline__ float exponent(float lse, float maximum) {
  return lse > -1.e29f ? exp2f(lse - maximum) : 0.f;
}

__global__ __launch_bounds__(kThreads, 1)
void attention_kernel(SparseMlaDecodeParams p, int chunks_per_cta, int chunks,
                   int splits, __nv_bfloat16* partial, float* partial_lse) {
  extern __shared__ __align__(16) unsigned char shared[];
  auto& sm = *reinterpret_cast<SharedStorage*>(shared);
  const int batch = blockIdx.x, split = blockIdx.y;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const auto* q = p.query + static_cast<std::int64_t>(batch) * 64 * 512;
  const int begin = split * chunks_per_cta;
  const int end = min(chunks, begin + chunks_per_cta);

  for (int i = tid; i < 64 * 28; i += kThreads) {
    int row = i / 28, group = i % 28;
    float values[16];
#pragma unroll
    for (int j = 0; j < 16; ++j)
      values[j] = __bfloat162float(q[row * 512 + group * 16 + j]);
    quantize16(values, sm.q[row] + group * 8, sm.qs[row] + group);
  }
  if (tid < 64) {
    sm.maximum[tid] = kEmpty;
    sm.denominator[tid] = 0.f;
    sm.alpha[tid] = 0.f;
  }
  // Each warp owns seven 8-column non-RoPE tiles and one 8-column RoPE
  // tile for all 64 heads. These FP32 accumulators live across all chunks.
  float output[4][7][4] = {};
  float rope_output[4][4] = {};
  if (begin < end) prefetch(p, sm.raw[0], batch, begin);
  __pipeline_wait_prior(0);
  __syncthreads();

  for (int chunk = begin; chunk < end; ++chunk) {
    int buffer = (chunk - begin) & 1;
    const RawTile& raw = sm.raw[buffer];
    // Next raw tile can arrive while CuTe MMA/softmax/PV consume this one.
    if (chunk + 1 < end) prefetch(p, sm.raw[buffer ^ 1], batch, chunk + 1);

    float scores[4][4] = {};
#pragma unroll
    for (int k = 0; k < 7; ++k) {
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        nv_mma<224, 352, 28, 32>(sm.q[g * 16] + k * 32,
                                 raw.data[warp * 8] + k * 32,
                                 sm.qs[g * 16] + k * 4,
                                 raw.scale[warp * 8] + k * 4, scores[g], lane);
      }
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        bf_mma<512, 176, 1>(q + g * 16 * 512 + 448 + k * 16,
                            reinterpret_cast<const __nv_bfloat16*>(raw.data[warp * 8] + 224) + k * 16,
                            scores[g], lane);
      }
    }

    // CuTe's C layout has two rows/lane and two adjacent columns/row.
    // Four lanes reduce each row's eight candidates, then shared memory
    // combines the eight warps. Store exponent values in scores registers.
#pragma unroll
    for (int g = 0; g < 4; ++g) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        int row = g * 16 + result_row(lane, r * 2);
        bool v0 = raw.valid[warp * 8 + result_column(lane, r * 2)];
        bool v1 = raw.valid[warp * 8 + result_column(lane, r * 2 + 1)];
        float z0 = v0 ? scores[g][r * 2] * (p.softmax_scale * kLog2e) : kEmpty;
        float z1 = v1 ? scores[g][r * 2 + 1] * (p.softmax_scale * kLog2e) : kEmpty;
        float m = fmaxf(z0, z1);
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
        float w0 = v0 ? exp2f(z0 - m) : 0.f;
        float w1 = v1 ? exp2f(z1 - m) : 0.f;
        scores[g][r * 2] = w0;
        scores[g][r * 2 + 1] = w1;
        float sum = w0 + w1;
        sum += __shfl_xor_sync(0xffffffff, sum, 1);
        sum += __shfl_xor_sync(0xffffffff, sum, 2);
        if ((lane & 3) == 0) {
          sm.local_max[row][warp] = m;
          sm.local_sum[row][warp] = sum;
        }
      }
    }
    __syncthreads();
    if (tid < 64) {
      float block_max = kEmpty;
#pragma unroll
      for (int w = 0; w < 8; ++w) block_max = fmaxf(block_max, sm.local_max[tid][w]);
      float block_sum = 0.f;
#pragma unroll
      for (int w = 0; w < 8; ++w)
        block_sum += sm.local_sum[tid][w] * exp2f(sm.local_max[tid][w] - block_max);
      float old = sm.maximum[tid], next = fmaxf(old, block_max);
      float alpha = old > -1.e29f ? exp2f(old - next) : 0.f;
      sm.denominator[tid] = alpha * sm.denominator[tid] + block_sum * exp2f(block_max - next);
      sm.maximum[tid] = next;
      sm.alpha[tid] = alpha;
    }
    __syncthreads();
#pragma unroll
    for (int g = 0; g < 4; ++g) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int row = g * 16 + result_row(lane, i);
        int col = warp * 8 + result_column(lane, i);
        float w = scores[g][i] * exp2f(sm.local_max[row][warp] - sm.maximum[row]);
        sm.weight[row][col] = __float2bfloat16_rn(w);
        rope_output[g][i] *= sm.alpha[row];
#pragma unroll
        for (int v = 0; v < 7; ++v) output[g][v][i] *= sm.alpha[row];
      }
    }
    __syncthreads();

    // Distinct W/P buffers avoid an in-place float-to-FP4 read/write race.
    int prow = tid / 4, pg = tid % 4;
    float weights[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) weights[j] = __bfloat162float(sm.weight[prow][pg * 16 + j]);
    quantize16(weights, sm.p[prow] + pg * 8, sm.ps[prow] + pg);
    prepare_v(raw, sm);
    __syncthreads();

#pragma unroll
    for (int v = 0; v < 7; ++v) {
      int dim = (v * 8 + warp) * 8;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        // Match the document's FP32 chunk product followed by FP32 addition.
        float contribution[4] = {};
        nv_mma<32, 32, 4, 4>(sm.p[g * 16], sm.vt[dim], sm.ps[g * 16],
                              sm.vts[dim], contribution, lane);
#pragma unroll
        for (int i = 0; i < 4; ++i) output[g][v][i] += contribution[i];
      }
    }
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      float contribution[4] = {};
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        bf_mma<64, 1, 176>(sm.weight[g * 16] + k * 16,
                            reinterpret_cast<const __nv_bfloat16*>(raw.data[k * 16] + 224) + warp * 8,
                            contribution, lane);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) rope_output[g][i] += contribution[i];
    }
    // Every current-buffer reader and asynchronous next-buffer writer has
    // finished before either raw buffer can be consumed/recycled.
    __pipeline_wait_prior(0);
    __syncthreads();
  }

  if (tid < 64) {
    float denom = sm.denominator[tid];
    float lambda = denom > 0.f ? sm.maximum[tid] + log2f(denom) : kEmpty;
    float factor = denom > 0.f ? 1.f / denom : 0.f;
    auto index = static_cast<std::int64_t>(batch) * 64 + tid;
    if (splits > 1) {
      partial_lse[index * splits + split] = lambda;
    } else {
      float sink = p.sink ? p.sink[tid] * kLog2e : -CUDART_INF_F;
      float maximum = fmaxf(lambda, sink);
      float mass = exponent(lambda, maximum);
      float sink_mass = p.sink ? exp2f(sink - maximum) : 0.f;
      float total = mass + sink_mass;
      factor *= total > 0.f ? mass / total : 0.f;
      p.lse[index] = total > 0.f ? (maximum + log2f(total)) * p.lse_scale : -CUDART_INF_F;
    }
    sm.alpha[tid] = factor;
  }
  __syncthreads();
#pragma unroll
  for (int g = 0; g < 4; ++g) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int row = g * 16 + result_row(lane, i);
      auto index = static_cast<std::int64_t>(batch) * 64 + row;
      auto* dest = splits > 1 ? partial + (index * splits + split) * 512 : p.output + index * 512;
#pragma unroll
      for (int v = 0; v < 7; ++v)
        dest[(v * 8 + warp) * 8 + result_column(lane, i)] =
            __float2bfloat16_rn(output[g][v][i] * sm.alpha[row]);
      dest[448 + warp * 8 + result_column(lane, i)] =
          __float2bfloat16_rn(rope_output[g][i] * sm.alpha[row]);
    }
  }
}

// One CTA/head. Sink is introduced exactly once, after all split LSE values
// are known. Normalized BF16 split outputs intentionally retain the tex's
// intermediate rounding; final accumulation and LSE are FP32.
__global__ void merge_kernel(SparseMlaDecodeParams p, int splits,
                             const __nv_bfloat16* partial, const float* partial_lse) {
  int head = blockIdx.x % 64;
  std::int64_t index = blockIdx.x;
  __shared__ float maximum, denominator;
  if (threadIdx.x == 0) {
    float sink = p.sink ? p.sink[head] * kLog2e : -CUDART_INF_F;
    float m = fmaxf(kEmpty, sink);
    for (int s = 0; s < splits; ++s) m = fmaxf(m, partial_lse[index * splits + s]);
    float d = p.sink ? exp2f(sink - m) : 0.f;
    for (int s = 0; s < splits; ++s) d += exponent(partial_lse[index * splits + s], m);
    maximum = m;
    denominator = d;
    p.lse[index] = d > 0.f ? (m + log2f(d)) * p.lse_scale : -CUDART_INF_F;
  }
  __syncthreads();
  for (int col = threadIdx.x; col < 512; col += blockDim.x) {
    float out = 0.f;
    for (int s = 0; s < splits; ++s) {
      float weight = exponent(partial_lse[index * splits + s], maximum);
      out += weight * __bfloat162float(partial[(index * splits + s) * 512 + col]);
    }
    p.output[index * 512 + col] = __float2bfloat16_rn(denominator > 0.f ? out / denominator : 0.f);
  }
}

}  // namespace sm120_nvfp4::sparse_mla

namespace sm120_nvfp4 {

int sparse_mla_chunks_per_cta(int batch, int swa, int compressed, int requested) {
  // A bounded public candidate capacity also keeps grid/workspace arithmetic
  // inside the supported int32 range. Typical CSA capacities are 128 and 512.
  if (batch <= 0 || batch > 1048576 || swa < 0 || compressed < 0 ||
      swa > 1048576 || compressed > 1048576 || requested < 0) return 0;
  int chunks = std::max(1, (swa + 63) / 64 + (compressed + 63) / 64);
  return requested ? std::min(requested, chunks) : (batch <= 64 ? std::min(9, chunks) : chunks);
}

int sparse_mla_split_count(int batch, int swa, int compressed, int requested) {
  int cpb = sparse_mla_chunks_per_cta(batch, swa, compressed, requested);
  if (cpb == 0) return 0;
  int chunks = std::max(1, (swa + 63) / 64 + (compressed + 63) / 64);
  return (chunks + cpb - 1) / cpb;
}

std::size_t sparse_mla_decode_workspace_size(int batch, int swa, int compressed, int requested) {
  int splits = sparse_mla_split_count(batch, swa, compressed, requested);
  if (splits <= 1) return 0;
  // Both arrays and their boundary are naturally 256-byte aligned.
  return static_cast<std::size_t>(batch) * 64 * splits * (512 * sizeof(__nv_bfloat16) + sizeof(float));
}

GemmStatus sparse_mla_decode_sm120(const SparseMlaDecodeParams& p,
                                 void* workspace, std::size_t workspace_bytes,
                                 cudaStream_t stream) {
  int cpb = sparse_mla_chunks_per_cta(p.batch, p.swa_candidates, p.compressed_candidates, p.chunks_per_cta);
  if (cpb == 0 || !p.query || !p.output || !p.lse || p.swa_pages < 0 || p.compressed_pages < 0 ||
      (p.swa_pages && !p.swa_cache) || (p.compressed_pages && !p.compressed_cache) ||
      (p.swa_candidates && !p.swa_indices) || (p.compressed_candidates && !p.compressed_indices) ||
      !(std::isfinite(p.softmax_scale) && p.softmax_scale > 0.f) ||
      !(std::isfinite(p.lse_scale) && p.lse_scale > 0.f) ||
      reinterpret_cast<std::uintptr_t>(p.query) % 16 ||
      reinterpret_cast<std::uintptr_t>(p.swa_cache) % 16 ||
      reinterpret_cast<std::uintptr_t>(p.compressed_cache) % 16) return GemmStatus::kInvalidArgument;
  int splits = sparse_mla_split_count(p.batch, p.swa_candidates, p.compressed_candidates, cpb);
  std::size_t required = sparse_mla_decode_workspace_size(p.batch, p.swa_candidates, p.compressed_candidates, cpb);
  if (required && (!workspace || workspace_bytes < required ||
                   reinterpret_cast<std::uintptr_t>(workspace) % 256)) return GemmStatus::kInsufficientWorkspace;
  int device;
  cudaDeviceProp properties{};
  if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&properties, device) != cudaSuccess)
    return GemmStatus::kCudaError;
  if (properties.major != 12 || properties.minor != 0) return GemmStatus::kUnsupportedDevice;
  if (cudaFuncSetAttribute(sparse_mla::attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           sizeof(sparse_mla::SharedStorage)) != cudaSuccess) return GemmStatus::kCudaError;
  auto* partial = static_cast<__nv_bfloat16*>(workspace);
  float* partial_lse = splits > 1 ? reinterpret_cast<float*>(
      partial + static_cast<std::size_t>(p.batch) * 64 * splits * 512) : nullptr;
  int chunks = (p.swa_candidates + 63) / 64 + (p.compressed_candidates + 63) / 64;
  sparse_mla::attention_kernel<<<dim3(p.batch, splits), 256, sizeof(sparse_mla::SharedStorage), stream>>>(
      p, cpb, chunks, splits, partial, partial_lse);
  if (cudaPeekAtLastError() != cudaSuccess) return GemmStatus::kCudaError;
  if (splits > 1) {
    sparse_mla::merge_kernel<<<p.batch * 64, 256, 0, stream>>>(p, splits, partial, partial_lse);
    if (cudaPeekAtLastError() != cudaSuccess) return GemmStatus::kCudaError;
  }
  return GemmStatus::kSuccess;
}

}  // namespace sm120_nvfp4
