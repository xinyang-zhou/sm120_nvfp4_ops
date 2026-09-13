#pragma once

#include <cstddef>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "sm120_nvfp4/gemm.hpp"

namespace sm120_nvfp4 {

// Scratch storage for dense prefill attention. The workspace contains FP32
// logits followed by a dynamically quantized NVFP4 probability matrix, its
// CUTLASS block-scale metadata, and one FP32 row-normalization correction.
std::size_t nvfp4_attention_workspace_size_sm120(
    int batch, int heads, int query_length, int kv_length);

// Dense prefill attention on SM120:
//
//   O = softmax(Q K^T * softmax_scale + causal_mask) V
//
// Q is packed [B,H,M,D/2], K is packed [B,H,N,D/2], and V_transposed is
// packed [B,H,Dv,N/2]. Each matrix owns a contiguous physical UE4M3 scale
// region in Sm1xxBlockScaledConfig<16> layout. Output is FP16 [B,H,M,Dv].
//
// The two matrix products execute with SM120 NVFP4 block-scaled MMA and FP32
// accumulation. Softmax is evaluated in FP32, then its probabilities are
// dynamically quantized to NVFP4 for the P @ V product. The reconstructed
// probability row sum is corrected after that MMA. A causal mask is
// suffix-aligned when M < N: query row r corresponds to KV position N-M+r.
GemmStatus nvfp4_attention_prefill_sm120(
    int batch, int heads, int query_length, int kv_length,
    int head_dim, int value_dim,
    const void* query, const void* key, const void* value_transposed,
    const void* query_scale, const void* key_scale,
    const void* value_scale,
    bool causal, float softmax_scale,
    half* output,
    void* workspace, std::size_t workspace_bytes,
    cudaStream_t stream = nullptr);

// Scratch storage for fused single-token dense decode. It holds grouped query
// scales plus output-sized split-K partials/LSE; full logits and probabilities
// are never placed in global memory. max_kv_length is the padded cache capacity
// and must be a multiple of 32.
std::size_t nvfp4_attention_decode_workspace_size_sm120(
    int batch, int query_heads, int kv_heads, int max_kv_length,
    int head_dim, int value_dim);

// Single-token dense decode with GQA/MQA support:
//
//   query                  [B,Hq,D/2]
//   key_cache              [B,Hkv,N,D/2]
//   value_cache_transposed [B,Hkv,Dv,N/2]
//   output                 [B,Hq,Dv]
//
// Hq must be divisible by Hkv. Consecutive groups of Hq/Hkv query heads map
// to one KV head. kv_lengths is an optional device int32 array [B]; nullptr
// means that all N cache positions are valid. Values are clamped to [0,N] by
// the softmax mask. Query scales keep the public per-query-head M=1 physical
// layout; decode repacks them into grouped M=Hq/Hkv scale tiles in workspace.
// QK, online softmax, tile-local probability quantization and NVFP4 PV execute
// in one kernel. Multiple sequence chunks are combined through their LSE.
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
    cudaStream_t stream = nullptr);

// Single-token paged decode with the same fused QK/softmax/PV schedule.
// Cache payloads are block-major and keep every KV head contiguous inside a
// physical block:
//
//   key_cache              [P,Hkv,S,D/2]
//   value_cache_transposed [P,Hkv,Dv,S/2]
//   block_table            [B,max_blocks] int32 device array
//   kv_lengths             [B] int32 device array
//
// P is the physical block count and S is 32, 64, or 128. Logical token t for
// request b is read from physical block block_table[b,t/S] at offset t%S.
// Each (physical block, KV head) owns an independent SFB scale region:
//
//   key_scale   P*Hkv * scale_b_elements(1, S,  D)
//   value_scale P*Hkv * scale_b_elements(1, Dv, S)
//
// Query scales and workspace use max_kv_length=max_blocks*S exactly as in the
// dense decode entry point. Full dense KV, logits, and probabilities are never
// materialized.
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
    cudaStream_t stream = nullptr);

}  // namespace sm120_nvfp4
