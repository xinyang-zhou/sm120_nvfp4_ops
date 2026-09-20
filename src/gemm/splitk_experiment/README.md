# Split-K GEMM experiment

This directory builds an isolated comparison between the existing Custom CuTe
GEMM and the experimental Split-K implementation. It does not modify or select
the default library path.

```bash
cmake -S src/gemm/splitk_experiment -B build-splitk \
  -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" \
  -DCUTLASS_ROOT="$CUTLASS_ROOT"
cmake --build build-splitk -j4

CUDA_VISIBLE_DEVICES=0 \
  ./build-splitk/sm120_nvfp4_splitk_benchmark \
  128 4096 8192 4 50 500
```

The reported Split-K latency includes both the partial GEMM and the FP32
reduction kernel. Workspace is caller-owned and allocated before timing.
