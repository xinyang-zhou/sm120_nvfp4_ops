# Performance

## Methodology

Unless noted otherwise, results were measured on an idle NVIDIA GeForce RTX 5090 with CUDA 13.2, driver 595.84, cuBLASLt 13.4 and CUTLASS 4.2.1.

The current GEMM benchmark compares three independent paths:

1. repository-owned Custom CuTe kernel;
2. CUTLASS `GemmUniversalAdapter` reference;
3. cuBLASLt with the fastest successfully timed heuristic candidate.

All paths:

- consume identical packed E2M1 inputs and UE4M3 physical scale buffers;
- accumulate in FP32 and output FP16;
- run on one CUDA stream and are measured with CUDA Events after warmup;
- validate every output element;
- report dense-equivalent `2*M*N*K` FLOP/s.

The CUTLASS and Custom CuTe outputs are row-major. cuBLASLt returns the equivalent column-major view; the benchmark accounts for this during validation.

## Custom CuTe single GEMM

Current measurements for `N=4096, K=8192`:

| M | Implementation | Latency | TFLOP/s | vs cuBLASLt | Validation |
|---:|---|---:|---:|---:|---:|
| 16 | Custom CuTe | 25.2 us | 42.5665 | 104.3358% | 0 mismatches |
| 16 | CUTLASS reference | 26.2 us | 41.0031 | 100.5037% | 0 mismatches |
| 16 | cuBLASLt id 70 | 26.3 us | 40.7976 | 100% | reference |
| 128 | Custom CuTe | 30.0 us | 286.7144 | 61.2723% | 0 mismatches |
| 128 | CUTLASS reference | 26.4 us | 325.4973 | 69.5604% | 0 mismatches |
| 128 | cuBLASLt id 70 | 18.4 us | 467.9348 | 100% | reference |
| 512 | Custom CuTe | 34.8 us | 987.0538 | 85.5611% | 0 mismatches |
| 512 | CUTLASS reference | 28.3 us | 1215.4651 | 105.3605% | 0 mismatches |
| 512 | cuBLASLt id 70 | 29.8 us | 1153.6246 | 100% | reference |

Measurement counts:

- M=16: 50 warmups, 500 iterations;
- M=128: 30 warmups, 300 iterations;
- M=512: 30 warmups, 200 iterations.

Interpretation:

- At M=16, the Custom CuTe path removes part of the generic adapter/scheduler overhead and reaches 104.34% of the selected cuBLASLt algorithm.
- At M=128, cuBLASLt selects a 12 MiB-workspace candidate and is substantially faster than both repository paths.
- At M=512, the CUTLASS reference remains strong while Custom CuTe reaches 85.56%; the current scalar/predicated epilogue and fixed one-CTA-per-output-tile scheduling are the primary optimization targets.
- Correctness is not inferred from matching aggregate statistics: every result element is compared, and Custom CuTe also matches the row-major CUTLASS reference exactly in these runs.

## Historical CUTLASS-only sweep

Before Custom CuTe was added, `nvfp4_gemm_sm120` referred to the CUTLASS Collective implementation. The earlier sweep below is retained for experiment history, but it must not be presented as Custom CuTe performance.

| M | CUTLASS Collective TFLOP/s | cuBLASLt TFLOP/s | Relative |
|---:|---:|---:|---:|
| 16 | 39.50 | 40.78 | 96.86% |
| 32 | 78.73 | 116.20 | 67.75% |
| 64 | 157.94 | 232.78 | 67.85% |
| 128 | 316.03 | 465.36 | 67.91% |
| 256 | 629.36 | 697.95 | 90.17% |
| 512 | 1196.34 | 1157.07 | 103.39% |
| 1024 | 1081.22 | 1195.05 | 90.48% |
| 2048 | 1081.96 | 1243.43 | 87.01% |
| 4096 | 1257.44 | 1294.77 | 97.12% |

Consequently, the historical M=512 result means “configured CUTLASS Collective exceeded the selected cuBLASLt heuristic on this shape,” not “the repository-owned Custom CuTe kernel exceeded cuBLASLt.”

## Dense and paged decode attention

The decode benchmark compares the repository's fused single-token dense and
paged KV paths. Both paths use native SM120 NVFP4 MMA for QK and PV, online
FP32 softmax, the same random packed E2M1 logical tensors, and preallocated
output/workspace buffers. The paged input is the dense logical cache split
into physical pages and addressed through an `int32` block table. Before
timing, every paged output element is checked against the dense result with
zero tolerance.

Configuration: RTX 5090 GPU 0, driver 595.84, CUDA Toolkit 13.2, PyTorch CUDA
12.8, `Hq=32`, `Hkv=8`, `N=1024`, `D=Dv=128`, 30 warmups and 300 CUDA Event
iterations.

| Batch | Page size | Dense latency | Paged latency | Paged / Dense | Max diff |
|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 18.28 us | 31.39 us | 1.72x | 0 |
| 1 | 64 | 18.10 us | 30.46 us | 1.68x | 0 |
| 1 | 128 | 17.96 us | 29.94 us | 1.67x | 0 |
| 8 | 32 | 22.89 us | 109.81 us | 4.80x | 0 |
| 8 | 64 | 22.90 us | 107.45 us | 4.69x | 0 |
| 8 | 128 | 22.87 us | 106.43 us | 4.65x | 0 |

The dense path can issue contiguous TMA transfers for K/V. The initial paged
implementation instead gathers packed bytes and scale factors through the
block table into the MMA shared-memory layout. The measured gap therefore
quantifies the current general gather overhead; it is not caused by
materializing a dense cache or full logits/probabilities. Page-size
specialization, cached physical-page lookup, and wider layout-aware memory
transactions are the next optimization targets.

Raw results:
[attention_decode_rtx5090_2026-09-13.json](../benchmarks/results/attention_decode_rtx5090_2026-09-13.json).

```bash
PYTHONPATH="$PWD/build/python" \
python benchmarks/benchmark_attention_decode.py \
  --batches 1,8 --block-sizes 32,64,128 \
  --warmup 30 --iterations 300 \
  --output benchmarks/results/attention_decode.json
```

## Generated instruction verification

The repository-owned kernel symbol appears as `sm120_nvfp4::cute_gemm_detail::nvfp4_gemm_kernel<...>`. Disassembly of the linked test binary shows `OMMA.SF` inside that function:

```text
Function : ...cute_gemm_detail...nvfp4_gemm_kernel...
OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X
```

This verifies that the custom path executes native SM120 block-scaled Tensor Core instructions rather than dequantizing to a wider type.

## Grouped GEMM

The native cuBLASLt pointer-array grouped layout probe returns:

```text
heuristic status=7 count=0
```

For this CUDA/cuBLASLt release, the tested NVFP4 grouped configuration has no usable native heuristic. The available baseline is therefore explicitly a loop baseline: one native GEMM launch per expert.

For equal per-expert `M=16`, `N=4096`, `K=8192`:

| Groups | Total M | Persistent grouped ms | cuBLASLt loop ms | Relative throughput |
|---:|---:|---:|---:|---:|
| 1 | 16 | 0.0742 | 0.0380 | 51.22% |
| 2 | 32 | 0.0891 | 0.0747 | 83.84% |
| 4 | 64 | 0.0770 | 0.1442 | 187.23% |
| 8 | 128 | 0.1516 | 0.2867 | 189.08% |
| 16 | 256 | 0.2524 | 0.5746 | 227.64% |
| 32 | 512 | 0.4322 | 1.1460 | 265.16% |

These numbers include the custom grouped operator's per-call metadata/TensorMap setup. The loop excludes output concatenation but necessarily contains G GEMM launches. This answers “persistent grouped launch versus a loop of native GEMMs”; it is not a comparison with a native cuBLASLt grouped NVFP4 kernel.

## DeepEP-compatible Fused MoE hand-off

The repository now exposes an `expert_moe` path for activations that a
dispatcher has already arranged in expert-major order. It consumes the
dispatcher-provided `seqlens/cu_seqlens`, dynamically quantizes BF16
activations to NVFP4, runs gate/up and down Persistent Grouped GEMMs, and
returns one result per expanded route for the dispatcher to combine.

The comparison uses the same reference Dispatch/Combine and the same expanded
expert-major input for both paths:

- **direct**: reuse expert layout and a caller-owned workspace, then call
  `expert_moe`;
- **reroute**: pass the already grouped input through the original
  `fused_moe`, which repeats local count/gather/reduce and creates temporary
  tensors.

Configuration: 2 x RTX 5090 (physical GPU 1,2), cross-NUMA `SYS` PCIe path,
256 tokens/rank, hidden 4096, intermediate 2048, top-k 2, 32 experts, 10
warmups and 100 alternating paired measurements. Times are the slower rank's
wall-clock P50.

| Routing | Expert M range | Direct compute | Reroute compute | Compute speedup | Direct end-to-end | Reroute end-to-end | E2E speedup | Max diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Balanced | 32--32 | 0.2365 ms | 0.3002 ms | 1.27x | 0.8187 ms | 0.8496 ms | 1.04x | 0 |
| 80% rank-skewed | 0--408 | 0.1506 ms | 0.2049 ms | 1.36x | 0.7905 ms | 0.8175 ms | 1.03x | 0 |

This is an integration/control-path benchmark using a transparent
`torch.distributed` NCCL reference transport. It is **not** a native DeepEP
kernel or NVLink/RDMA bandwidth result. The useful result is that consuming
the dispatcher layout directly reduces the local Expert compute path while
keeping Dispatch/Combine semantics unchanged. Raw results:
[balanced](../benchmarks/results/deepep_handoff_balanced_rtx5090_2026-09-11.json)
and
[skewed](../benchmarks/results/deepep_handoff_skewed_rtx5090_2026-09-11.json).
The direct/reroute equivalence test also passes CUDA Compute Sanitizer memcheck
with [0 errors](../benchmarks/results/expert_moe_memcheck_rtx5090_2026-09-11.txt).

```bash
CUDA_VISIBLE_DEVICES=1,2 NCCL_IB_DISABLE=1 \
PYTHONPATH="$PWD/build/python" \
torchrun --standalone --nproc-per-node=2 \
  benchmarks/benchmark_deepep_handoff.py \
  --tokens 256 --hidden 4096 --intermediate 2048 \
  --experts 32 --topk 2 --routing balanced \
  --warmup 10 --iterations 100
```

Use `--routing skewed` for the second row.

## Reproduction

```bash
./scripts/build.sh
CUDA_VISIBLE_DEVICES=0 ./scripts/benchmark.sh \
  16 4096 8192 --warmup 50 --iterations 500 --heuristics 16
```

Representative M=512 command:

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/benchmark.sh \
  512 4096 8192 --warmup 30 --iterations 200 --heuristics 16
```

Always record clocks/power mode, driver, CUDA, cuBLASLt, CUTLASS commit and GPU occupancy conditions when publishing results.
