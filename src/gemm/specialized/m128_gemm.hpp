#pragma once

#include <cstddef>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "sm120_nvfp4/gemm.hpp"

namespace sm120_nvfp4::gemm_specialized::m128 {

// M=128 Split-K specialization selected by the default GEMM dispatcher.
// This remains an internal implementation detail rather than a public backend.
// It supports groups=1 and split_k=4; other values are rejected internally.
//
// Workspace layout is split-major FP32:
//   [split_k, groups, m, n]
//
// A return value of zero means that the arguments are invalid.
std::size_t workspace_size(
    int groups, int m, int n, int k, int split_k);

GemmStatus launch(
    int groups, int m, int n, int k, int split_k,
    const void* a, const void* b,
    const void* sfa, const void* sfb,
    half* output,
    void* workspace, std::size_t workspace_bytes,
    cudaStream_t stream = nullptr);

}  // namespace sm120_nvfp4::gemm_specialized::m128
