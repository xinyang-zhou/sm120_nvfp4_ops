# Performance

## Methodology

Unless noted otherwise, results were measured on an idle NVIDIA GeForce RTX 5090 with CUDA 13.2, driver 595.84, cuBLASLt 13.4 and CUTLASS 4.2.1.

Unless a section explicitly says otherwise, latency in this document is
standalone operator/kernel latency measured without a profiler. It must not be
read as model-level or end-to-end inference latency. The exact evidence status
for every headline table is tracked in
[the result index](../benchmarks/results/README.md).

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
| 128 | Custom CuTe | 26.2 us | 327.7392 | 70.2297% | 0 mismatches |
| 128 | CUTLASS reference | 26.4 us | 325.7989 | 69.8139% | 0 mismatches |
| 128 | cuBLASLt id 70 | 18.4 us | 466.6673 | 100% | reference |
| 512 | Custom CuTe | 28.4 us | 1208.3931 | 105.4995% | 0 mismatches |
| 512 | CUTLASS reference | 28.2 us | 1216.8333 | 106.2364% | 0 mismatches |
| 512 | cuBLASLt id 70 | 30.0 us | 1145.4020 | 100% | reference |

These GEMM rows are retained historical measurements. They predate the
structured writer, so the original JSON/CSV files are not present in the
repository. Re-run the commands below and check in their generated artifacts
before using the numbers as fully traceable performance evidence.

Measurement counts:

- M=16: 50 warmups, 500 iterations;
- M=128: 50 warmups, 500 iterations;
- M=512: 50 warmups, 300 iterations.

Interpretation:

- At M=16, the Custom CuTe path keeps its low-overhead predicated scalar
  epilogue and reaches 104.34% of the selected cuBLASLt algorithm.
- At M=128, the TMA-store epilogue reduces Custom CuTe latency from the prior
  29.9 us scalar-store baseline to 26.2 us. cuBLASLt still wins by selecting a
  12 MiB-workspace candidate.
- At M=512, TMA store reduces Custom CuTe latency from 35.0 us to 28.4 us,
  bringing it within 0.7% of the CUTLASS reference and to 105.50% of the
  selected cuBLASLt algorithm.
- Correctness is not inferred from matching aggregate statistics: every result element is compared, and Custom CuTe also matches the row-major CUTLASS reference exactly in these runs.

The runtime keeps scalar stores for `M < 64`: staging an entire 128x128 tile
made M=16 about 2% slower and M=32 about 1.6% slower. At M=64, TMA measured
25.9 us versus 26.9 us for scalar stores, so 64 rows is the current measured
crossover. This threshold is specific to the fixed 128x128 tile and should be
revisited together with future shape autotuning.

## Default fixed-M dispatcher

The public default `nvfp4_gemm_sm120`/Python `gemm` entry points now select
two repository-owned fixed-M implementations. The explicit
`nvfp4_cute_gemm_sm120`/Python `cute_gemm` entry points remain the generic
baseline.

RTX 5090 CUDA Event results for `N=4096, K=8192`, 500 warmups and 500 timed
iterations:

| M | Selected path | Workspace | Generic CuTe | Dispatched total | Throughput | Speedup | Validation |
|---:|---|---:|---:|---:|---:|---:|---:|
| 128 | Split-K=4 | 8 MiB | 26.211 us | 13.027 us | 659.41 TFLOP/s | 2.012x | 0 mismatches |
| 256 | Split-K=2 | 8 MiB | 26.325 us | 18.103 us | 949.03 TFLOP/s | 1.454x | 0 mismatches |

The M=128 result was stable at approximately 13.02 us in four of five repeated
runs; one cold/outlier run measured 15.35 us. Three final M=256 runs measured
18.103--18.106 us. All reported comparisons validated every output element
against the generic path with zero absolute and relative error.

Nsight Systems separated the final M=256 implementation into a 15.739 us
partial GEMM and a 2.188 us reduction. The checked-in benchmark measures the
complete two-kernel path and prints the selected dispatcher path:

```bash
CUDA_VISIBLE_DEVICES=0 ./build/benchmark_gemm_specialized \
  128 4096 8192 500 500
CUDA_VISIBLE_DEVICES=0 ./build/benchmark_gemm_specialized \
  256 4096 8192 500 500
```

These measurements were collected interactively during specialization work;
the profiler reports are intentionally not part of the source tree. They
should be rerun into a structured artifact before being cited as fully
traceable external evidence.

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

## DS-V4 CSA sparse MLA decode and prefill

Measured 2026-09-27 on one RTX 5090, using the retained warp-specialized
implementation (eight compute plus four IO warps, two raw-KV stages).
The same main kernel serves all three batches; default chunk scheduling
uses CPB=9/two splits at B=64 and CPB=10/one split at B=512/1024.

| Batch | This library P50 / μs | FlashInfer P50 / μs | Speedup (FI/ours) | Latency reduction |
|---|---:|---:|---:|---:|
| 64 | 73.728 | 112.640 | 1.528× | 34.55% |
| 512 | 280.576 | 464.864 | 1.657× | 39.64% |
| 1024 | 475.136 | 801.792 | 1.688× | 40.74% |

Conditions and scope:

- CUDA 13.2 compiler, PyTorch 2.7.0+cu128 (PyTorch reports CUDA 12.8),
  CUTLASS 4.2.1; FlashInfer 0.7.0 at revision
  `ea728cb558c32a3c58ec8fbd5a154ff676b9ab70`.
  This measurement used an attention-only extension in that environment;
  the full repository's GEMM/MoE bindings require a newer PyTorch build with
  `torch.float4_e2m1fn_x2`. Do not treat 2.7 as a full-build requirement.
- CSA single-token decode with 32768 history tokens per request, 64 query
  heads, D=448+64, 128 SWA and 512 selected compressed candidates. Synthetic
  unit-RMS inputs, random sink, seed43; random unique chronological selection.
- BF16 Q/output, NVFP4 non-RoPE and BF16 RoPE QK/PV. Baseline is the public
  sparse NVFP4 API with its own planner, not an artificially matched-CPB path.
- Twenty warmups, five rounds of 100 CUDA Graph samples per backend/batch,
  pooled P50; fixed inputs with 256 MiB L2 disturbance before each call and
  outside the event interval. Backend measurement order alternates by batch.
- No profiler; clocks were not locked. Background processes were allowed,
  but monitoring detected no foreign compute process during this run.
- Includes online Q/P/V conversion and any required split merge. Excludes
  cache packing, compressor/indexer/selection, RoPE, projections, model
  scheduling, setup, JIT and reference calculations.

The sparse MLA suite passed 25 tests including FlashInfer integration. In
this paired benchmark all three cases passed; output RMSE versus FlashInfer
was 2.02e-6, 1.90e-6 and 1.89e-6, with no nonfinite outputs. Public FlashInfer
does not return LSE, so this library's LSE was checked against the independent
reference instead. These checks do not establish model-quality equivalence.
Prefill shares the kernel and is correctness-tested, but no prefill speedup
is claimed. A separate sanitizer run of this warp-specialized revision is
still outstanding.

Measured source identifiers (SHA256):

- `src/attention/sparse_mla/decode.cu`:
  `236f2653a7f1e9f76123cb386c1dec814354f943fbb803075e6be0420ad09602`
- `src/attention/sparse_mla/mma.cuh`:
  `90dbd35ee72268766391f809472d962a0f7f6974a211bfc667e541f4cf487dd4`

Reproduce after building the extension (the output must not already exist):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/build/python" \
python3 benchmarks/benchmark_attention_ops.py \
  --modes decode --kinds csa --batches 64 512 1024 \
  --context-length 32768 --warmup 20 --rounds 5 --samples 100 --seed 43 \
  --flashinfer --flashinfer-dispatch public \
  --output temp/attention_flashinfer_new.json
```

Raw results, NCU reports and internal optimization notes are intentionally
kept locally under ignored `temp/`, not checked in. Only the reviewed summary,
measurement conditions and source identifiers are published. See
[Sparse MLA](SPARSE_MLA.md) for the interface and validation commands.

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

The grouped table likewise predates the structured writer and currently has no
checked-in raw JSON/CSV. `benchmark_grouped_gemm_vs_cublaslt` now reproduces
the same comparison scope: one persistent grouped invocation versus a timed
loop containing one cuBLASLt launch per expert, with elementwise validation.

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

This optional two-GPU integration benchmark uses a transparent
`torch.distributed` NCCL reference transport. It is **not** a native DeepEP
kernel or NVLink/RDMA bandwidth result. Historical hand-off timing tables and
raw artifacts have been removed from the public snapshot; the interface and
reproduction script remain. The current headline comparison is sparse
attention versus FlashInfer above.

```bash
CUDA_VISIBLE_DEVICES=1,2 NCCL_IB_DISABLE=1 \
PYTHONPATH="$PWD/build/python" \
torchrun --standalone --nproc-per-node=2 \
  benchmarks/integration/benchmark_deepep_handoff.py \
  --tokens 256 --hidden 4096 --intermediate 2048 \
  --experts 32 --topk 2 --routing balanced \
  --warmup 10 --iterations 100 \
  > temp/deepep_handoff_balanced_new.json
```

Create `temp/` first. Use `--routing skewed` and a different output filename
for skewed routing. To check the direct/reroute interface with memcheck:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD/build/python" \
compute-sanitizer --tool memcheck \
  python -m unittest \
  tests.python.test_fused_moe.FusedMoeTest.test_expert_major_handoff_matches_reroute
```

## Reproduction

```bash
./scripts/build.sh
CUDA_VISIBLE_DEVICES=0 ./scripts/benchmark.sh \
  16 4096 8192 --warmup 50 --iterations 500 --heuristics 16 \
  --json benchmarks/results/gemm_rtx5090_YYYY-MM-DD.json \
  --csv benchmarks/results/gemm_rtx5090_YYYY-MM-DD.csv
```

Representative M=512 command:

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/benchmark.sh \
  512 4096 8192 --warmup 50 --iterations 300 --heuristics 16 \
  --json benchmarks/results/gemm_m512_rtx5090_YYYY-MM-DD.json \
  --csv benchmarks/results/gemm_rtx5090_YYYY-MM-DD.csv
```

Representative Grouped GEMM command:

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/benchmark_grouped.sh \
  4 16 4096 8192 --warmup 50 --iterations 500 --heuristics 16 \
  --json benchmarks/results/grouped_gemm_g4_rtx5090_YYYY-MM-DD.json \
  --csv benchmarks/results/grouped_gemm_rtx5090_YYYY-MM-DD.csv
```

`--json` writes one complete run and `--csv` appends one row, so a suite can
share one CSV while keeping an immutable JSON per shape. The benchmark records
the GPU, CUDA runtime/driver-API version, cuBLASLt version, stream, timing parameters,
input distribution, selected heuristic/workspace, correctness and command.
CUTLASS source revision, clocks/power mode and whether the GPU was otherwise
idle remain external experiment metadata and must be recorded alongside the
result files before publication.
