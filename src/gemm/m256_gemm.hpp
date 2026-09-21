#pragma once

#include <cstddef>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "sm120_nvfp4/gemm.hpp"

namespace sm120_nvfp4::m256_experiment {

// Specialized experimental path for M=256 and split_k=2.  It intentionally
// lives outside the public API and is not selected by the default GEMM
// dispatcher.  Other M or split_k values are rejected.
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

}  // namespace sm120_nvfp4::m256_experiment
