# Benchmark result index

This directory is the evidence index for performance claims in the repository.
A documentation table is not itself a raw benchmark artifact. `verified`
below means that a checked-in machine-readable result contains the measurement,
configuration and correctness outcome; `rerun required` means that the number
is historical and must not be presented as fully traceable evidence yet.

| Claim | Documentation | Raw artifact | Reproduction | Status |
|---|---|---|---|---|
| Single GEMM, M=16/128/512 | [Performance](../../docs/PERFORMANCE.md#custom-cute-single-gemm) | Not retained by the original run | Commands below | rerun required |
| Default fixed-M GEMM, M=128/256 | [Performance](../../docs/PERFORMANCE.md#default-fixed-m-dispatcher) | Interactive reports intentionally not retained | Commands below | rerun required |
| M=512 scalar 35.0 us to TMA 28.4 us | [Performance](../../docs/PERFORMANCE.md#custom-cute-single-gemm) | Not retained; scalar code is commit `4f227c7`, TMA code is commit `51fde90` | Benchmark both revisions under identical conditions | rerun required |
| Persistent Grouped GEMM, 1--32 groups | [Performance](../../docs/PERFORMANCE.md#grouped-gemm) | Not retained by the original run | Commands below | rerun required |
| Balanced Expert-major handoff | [Performance](../../docs/PERFORMANCE.md#deepep-compatible-fused-moe-hand-off) | [`deepep_handoff_balanced_rtx5090_2026-09-11.json`](deepep_handoff_balanced_rtx5090_2026-09-11.json) | [Performance command](../../docs/PERFORMANCE.md#deepep-compatible-fused-moe-hand-off) | verified |
| Skewed Expert-major handoff | [Performance](../../docs/PERFORMANCE.md#deepep-compatible-fused-moe-hand-off) | [`deepep_handoff_skewed_rtx5090_2026-09-11.json`](deepep_handoff_skewed_rtx5090_2026-09-11.json) | Use the same command with `--routing skewed` | verified |
| Expert-major memcheck | [Performance](../../docs/PERFORMANCE.md#deepep-compatible-fused-moe-hand-off) | [`expert_moe_memcheck_rtx5090_2026-09-11.txt`](expert_moe_memcheck_rtx5090_2026-09-11.txt) | [Performance command](../../docs/PERFORMANCE.md#deepep-compatible-fused-moe-hand-off) | verified |

## Single GEMM rerun

Use a separate JSON file per shape and one append-only CSV for the sweep:

```bash
mkdir -p benchmarks/results
for m in 16 128 512; do
  iterations=500
  if [[ "$m" == 512 ]]; then iterations=300; fi
  CUDA_VISIBLE_DEVICES=0 ./scripts/benchmark.sh \
    "$m" 4096 8192 --warmup 50 --iterations "$iterations" --heuristics 16 \
    --json "benchmarks/results/gemm_m${m}_rtx5090_YYYY-MM-DD.json" \
    --csv benchmarks/results/gemm_rtx5090_YYYY-MM-DD.csv
done
```

Do not overwrite the historical table merely because one rerun differs. Keep
the new raw files, record clocks/power/idle conditions, then update the table
with the new run date and explain the difference.

To validate the default selector against the explicit generic baseline:

```bash
for m in 128 256; do
  CUDA_VISIBLE_DEVICES=0 ./build/benchmark_gemm_specialized \
    "$m" 4096 8192 500 500
done
```

## Grouped GEMM rerun

The baseline below is explicitly a loop of independent cuBLASLt GEMMs. The
native pointer-array capability probe remains a separate command and must not
be conflated with this loop baseline.

```bash
for groups in 1 2 4 8 16 32; do
  CUDA_VISIBLE_DEVICES=0 ./scripts/benchmark_grouped.sh \
    "$groups" 16 4096 8192 --warmup 50 --iterations 500 --heuristics 16 \
    --json "benchmarks/results/grouped_gemm_g${groups}_rtx5090_YYYY-MM-DD.json" \
    --csv benchmarks/results/grouped_gemm_rtx5090_YYYY-MM-DD.csv
done

CUDA_VISIBLE_DEVICES=0 ./build/probe_cublaslt_grouped
```

Each structured record includes the command, GPU identity, CUDA runtime and
driver-API versions, cuBLASLt version, timing parameters, input generation,
heuristic search, workspace, latency, throughput and correctness. CUTLASS
revision and clock/power/idle state must be captured alongside the files until
they are emitted automatically by the benchmark environment collector.
