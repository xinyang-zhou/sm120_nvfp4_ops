# Benchmark result index

This directory is the evidence index for performance claims in the repository.
A documentation table is not itself a raw benchmark artifact. Raw local
measurements are kept under ignored `temp/`, because they can contain personal
paths, device IDs and worktree details. This directory keeps only the public
evidence index. `locally validated` means measured and correctness-checked,
with raw data retained privately; it does not mean raw evidence is checked in.
`rerun required` identifies older historical claims without retained evidence.

| Claim | Documentation | Raw artifact | Reproduction | Status |
|---|---|---|---|---|
| Single GEMM, M=16/128/512 | [Performance](../../docs/PERFORMANCE.md#custom-cute-single-gemm) | Not retained by the original run | Commands below | rerun required |
| Default fixed-M GEMM, M=128/256 | [Performance](../../docs/PERFORMANCE.md#default-fixed-m-dispatcher) | Interactive reports intentionally not retained | Commands below | rerun required |
| M=512 scalar 35.0 us to TMA 28.4 us | [Performance](../../docs/PERFORMANCE.md#custom-cute-single-gemm) | Not retained; scalar code is commit `4f227c7`, TMA code is commit `51fde90` | Benchmark both revisions under identical conditions | rerun required |
| Persistent Grouped GEMM, 1--32 groups | [Performance](../../docs/PERFORMANCE.md#grouped-gemm) | Not retained by the original run | Commands below | rerun required |
| CSA sparse decode vs FlashInfer, B=64/512/1024 | [Performance](../../docs/PERFORMANCE.md#ds-v4-csa-sparse-mla-decode-and-prefill) | Private latest data in ignored `temp/`; no public raw artifact | Command and source hashes in Performance | locally validated, 2026-09-27 |

Old DeepEP-compatible timing artifacts are no longer published. The optional
integration benchmark remains available, without a current performance claim.
Never force-add raw result files without reviewing their metadata for privacy.

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
