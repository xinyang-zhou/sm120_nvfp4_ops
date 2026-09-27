#!/usr/bin/env bash
# Server only: correctness first, then a bounded operator benchmark matrix.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build}"
PYTHON_BIN="${PYTHON:-python3}"
if [[ $# -lt 1 || $# -gt 2 || ( $# -eq 2 && "$2" != "--flashinfer" ) ]]; then
  echo "Usage: bash scripts/benchmark_operators.sh NEW_OUTPUT_DIR [--flashinfer]" >&2
  exit 2
fi
OUTPUT_DIR="$1"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Output path exists; choose a new directory: $OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${BUILD_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
FLASHINFER_ARGS=()
if [[ $# -eq 2 ]]; then
  FLASHINFER_ARGS=(--flashinfer)
fi
git rev-parse HEAD > "$OUTPUT_DIR/commit.txt"
git status --short > "$OUTPUT_DIR/working_tree.txt"
git diff --binary > "$OUTPUT_DIR/tracked_changes.patch"
nvidia-smi > "$OUTPUT_DIR/nvidia-smi.txt"
"${BUILD_DIR}/test_sm120_nvfp4_gemm" 2>&1 | tee "$OUTPUT_DIR/cpp_correctness.log"
"${PYTHON_BIN}" scripts/validate_operators.py "${FLASHINFER_ARGS[@]}" \
  --output "$OUTPUT_DIR/correctness.json" 2>&1 | tee "$OUTPUT_DIR/correctness.log"

# Existing C++ benchmarks compare to CUTLASS/cuBLASLt and gate correctness.
# Repeat independently rather than treating a single aggregate as three rounds.
for round in 1 2 3; do
  for m in 16 128 256 512; do
    "${BUILD_DIR}/benchmark_gemm_vs_cublaslt" "$m" 4096 8192 \
      --warmup 50 --iterations 300 --heuristics 16 \
      --json "$OUTPUT_DIR/gemm_m${m}_r${round}.json" \
      2>&1 | tee "$OUTPUT_DIR/gemm_m${m}_r${round}.log"
  done
  for groups in 1 8 32; do
    "${BUILD_DIR}/benchmark_grouped_gemm_vs_cublaslt" "$groups" 16 4096 8192 \
      --warmup 50 --iterations 300 --heuristics 16 \
      --json "$OUTPUT_DIR/grouped_g${groups}_r${round}.json" \
      2>&1 | tee "$OUTPUT_DIR/grouped_g${groups}_r${round}.log"
  done
  for m in 128 256; do
    "${BUILD_DIR}/benchmark_gemm_specialized" "$m" 4096 8192 100 300 \
      2>&1 | tee "$OUTPUT_DIR/specialized_m${m}_r${round}.log"
  done
done
"${PYTHON_BIN}" benchmarks/benchmark_attention_ops.py "${FLASHINFER_ARGS[@]}" \
  --mixed-lengths --output "$OUTPUT_DIR/attention.json" \
  2>&1 | tee "$OUTPUT_DIR/attention.log"
"${PYTHON_BIN}" benchmarks/benchmark_sparse_mla_cache.py "${FLASHINFER_ARGS[@]}" \
  --output "$OUTPUT_DIR/cache.json" 2>&1 | tee "$OUTPUT_DIR/cache.log"
"${PYTHON_BIN}" benchmarks/benchmark_moe_ops.py \
  --output "$OUTPUT_DIR/moe.json" 2>&1 | tee "$OUTPUT_DIR/moe.log"
echo "Operator gates and benchmark sweep finished: $OUTPUT_DIR"
