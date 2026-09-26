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

}  // namespace sm120_nvfp4
