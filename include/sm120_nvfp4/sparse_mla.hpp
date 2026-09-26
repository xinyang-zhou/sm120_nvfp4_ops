#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include "sm120_nvfp4/gemm.hpp"

namespace sm120_nvfp4 {

// DS-V4 CSA decode/prefill: Hq=64, Hkv=1, D=448 non-RoPE + 64 RoPE.
// Cache format is FlashInfer DSV4 NVFP4 with page_size=64. A physical page
// contains 64*352 data bytes, THEN 64*32 scale bytes (not interleaved rows).
// A data row contains 224 E2M1 bytes followed by 64 BF16 RoPE values;
// its scale row contains 28 E4M3 scales and 4 padding bytes.
// Indices are flattened physical slots in their respective cache pool.
// Negative/out-of-range indices and positions beyond lengths are masked.
// R denotes independent query rows: B for decode, T for packed prefill.
struct SparseMlaCommonParams {
  const __nv_bfloat16* query = nullptr;  // [R,64,512], already post-RoPE
  const std::uint8_t* swa_cache = nullptr;
  const std::uint8_t* compressed_cache = nullptr;
  const std::int32_t* swa_indices = nullptr;         // [R,swa_candidates]
  const std::int32_t* compressed_indices = nullptr;  // [R,compressed_candidates]
  const std::int32_t* swa_lengths = nullptr;         // optional [R]
  const std::int32_t* compressed_lengths = nullptr;  // optional [R]
  const float* sink = nullptr;                       // optional [64], natural logits
  __nv_bfloat16* output = nullptr;                   // [R,64,512]
  float* lse = nullptr;                             // [R,64], base-2 * lse_scale
  int swa_pages = 0;
  int compressed_pages = 0;
  int swa_candidates = 128;
  int compressed_candidates = 512;
  float softmax_scale = 0.04419417382415922f;  // 1/sqrt(512)
  float lse_scale = 1.0f;                    // use ln(2) for natural LSE
};

struct SparseMlaDecodeParams : SparseMlaCommonParams {
  int batch = 0;
  int chunks_per_cta = 0;  // 0: 9 for B<=64, all chunks otherwise
};

struct SparseMlaPrefillParams : SparseMlaCommonParams {
  int num_queries = 0;  // packed T in [1,1048576], across one or more requests
};

// Same deterministic plan is used by launch and workspace sizing. An empty
// candidate set still launches one CTA/request so sink/LSE are well defined.
int sparse_mla_chunks_per_cta(int batch, int swa_candidates,
                            int compressed_candidates, int requested);
int sparse_mla_split_count(int batch, int swa_candidates,
                          int compressed_candidates, int chunks_per_cta);
std::size_t sparse_mla_decode_workspace_size(
    int batch, int swa_candidates, int compressed_candidates, int chunks_per_cta);

// Only split outputs (BF16) and split LSE (FP32) occupy global scratch.
// Inputs/output/workspace must not alias. Finite Q/cache values are required;
// sink may also be -infinity. Each input occurrence counts once, including
// repeated indices; selection, causal visibility, compression and RoPE are
// caller responsibilities. This is the attention core, not the indexer/block.
GemmStatus sparse_mla_decode_sm120(const SparseMlaDecodeParams& params,
                                 void* workspace, std::size_t workspace_bytes,
                                 cudaStream_t stream = nullptr);

// One CTA/query processes all candidate chunks and writes final output/LSE.
// No split, merge kernel, global scratch or internal device allocation.
// Each query supplies its own indices/lengths into the shared cache pools;
// the caller must encode request isolation and causal visibility in those
// lists. No request offsets or positions are needed by this fixed-index core.
// All common precision, masking and non-aliasing rules above also apply.
GemmStatus sparse_mla_prefill_sm120(const SparseMlaPrefillParams& params,
                                  cudaStream_t stream = nullptr);

}  // namespace sm120_nvfp4
