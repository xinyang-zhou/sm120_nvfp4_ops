# FlashInfer SM120 CSA decode baseline

`benchmark_flashinfer_csa_decode.py` measures the existing FlashInfer public API
for B=64, 256 and 1024. Run it on the GPU server from this project's checkout,
using the Python environment in which FlashInfer's upstream tests passed.
The project extension does not need to be built for this script.

The reviewed FlashInfer source revisions are:

- `37b4d30eac39b89f198b893dd11914bd76f5fcf8` (initial local reference).
- `ea728cb558c32a3c58ec8fbd5a154ff676b9ab70` (the user's server installation).

The sparse MLA API, private planner, NVFP4 cache layout, SM120 sparse kernels
and upstream NVFP4 test file are unchanged between these revisions. There is
no need to downgrade the server installation. The script checks the imported
source location, revision and tracked runtime-source changes, and records the
actual revision in its output. Keep that revision fixed for performance
comparisons. Additional revisions require source review because the harness
uses private plan inspection and cache decoding.

## Run

First validate the new harness with the smallest target batch:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_flashinfer_csa_decode.py \
  --batches 64 --context-length 32768
```

Then collect the three target batches:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_flashinfer_csa_decode.py \
  --batches 64 256 1024 --context-length 32768
```

Default output is a short terminal summary. No log or data file is created.
For an archived baseline, add `--output /path/to/new/baseline.json`; this saves
every timing sample, per-round medians, input conventions, memory measurements,
environment information, source hashes, dispatch details and sampled errors.
An existing file is never overwritten.

## Workload and scope

The shape follows the [DeepSeek-V4-Flash configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json):
64 query heads, one shared KV head, D=512 (448 non-RoPE + 64 RoPE), CSA compression
ratio 4, compressed Top-K 512 and SWA window 128. These shape constants are stored
in the script; it does not fetch a moving model configuration at runtime.

- Each request has one query token: `T=B`. The shape models a complete head set
  (TP=1), not a complete model running on one GPU.
- `--context-length` means history tokens **before** the current query. The
  query position is L, and the compressed cache contains L/4 completed entries.
  For this initial harness L must be a multiple of 256 and at least 2048.
- Each request owns disjoint cache pages. The SWA pool holds positions
  `[L-127, L]`, including the current query's KV. The compressed selection samples
  512 distinct entries from that request's completed blocks, in random order.
- Both pools use HND layout and page size 64. They are supplied in the public
  API's semantic order: SWA as the primary segment and compressed entries as
  the secondary segment. This order differs from some upstream microbenchmarks.
- The entire visible compressed cache is allocated. At B=1024, L=32768 the two
  packed caches occupy about 3.05 GiB. Larger contexts increase this footprint.
  Temporary BF16 data is created in chunks to bound setup memory.
- Q/KV vectors are synthetic unit-RMS inputs at the attention-core boundary;
  the sink is synthetic as well. There are no learned normalization weights,
  real model activations, actual compressor outputs or indexer-selected indices.
  Causal positions and shape agree with a CSA decode snapshot, while values and
  selection statistics remain synthetic. This is not model-quality validation.
- BF16 Q/output, NVFP4 non-RoPE KV and BF16 RoPE KV match this FlashInfer NVFP4
  path. NVFP4 KV remains a precision experiment relative to the official model's
  FP8/BF16 KV format.

## Measurement

All batches invoke `flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4` with
`backend="sparse"` and `kv_cache_format="nvfp4"`. The public planner chooses the
implementation. In particular, a name containing `prefill` at B=256/1024 still
represents the same single-token-per-request decode workload.

The default timing uses CUDA events around a CUDA Graph replay containing one
public API invocation. JIT, calibration, graph capture and warmup are excluded.
Each of three rounds collects 100 individual samples. P50/P95 are percentiles of
these event durations; per-round medians and the complete samples are available
with `--output`. This measures isolated GPU decode execution; replay omits Python
API overhead. `--eager` separately measures event intervals around eager public
calls, which can include GPU idle time caused by host submission.

A 256 MiB buffer is zeroed before each timed invocation, outside the event
interval, to disturb L2. The buffer's size and memory cost are reported. Set
`--l2-flush-mib 0` for a separate repeated-input, warm-cache experiment. Keep this
setting and graph mode identical when comparing implementations. The operation
is a controlled cache disturbance, not a guarantee about every physical cache
line or a simulation of the full model's memory traffic.

Timed work includes online Q/P quantization, selected-V conversion, attention
and any split merge. It excludes cache packing/updating, compression, index
selection, projections and the rest of the layer. `core_q/s` is B divided by the
median core latency, not serving throughput or model token throughput.

The output is checked for NaN/Inf. The last timed output must match a warmed
eager call exactly, and the selected plan must remain stable through measurement.
Reference diagnostics cover requests 0, B//2 and B-1 (duplicates removed):

1. FP32 attention over the original synthetic BF16 Q/KV, with the same sink.
2. FP32 attention over BF16 Q and dequantized NVFP4 KV, with the same sink.

Both report max absolute error, RMSE, relative L2 and mean per-head cosine.
The second comparison still includes differences from online Q/P quantization
and V tile conversion. These diagnostics report the baseline error; they do not
apply a model-quality acceptance threshold or replace the upstream correctness
tests. After the initial measurements, set the regression budget for subsequent
optimization. Peak PyTorch memory is reported separately for setup and timing;
these numbers include allocator-tracked overhead and are not whole-process GPU
memory. Fixed workspace capacity is 64 MiB; actual plan requirements are also
present in the JSON.

## Validation status

The user reported `84 passed in 2.32s` for the upstream
`tests/attention/test_sparse_mla_sm120_dsv4_nvfp4.py` on the server. The first
harness run stopped at the revision check, before any GPU benchmark, because
the server uses `ea728cb558c32a3c58ec8fbd5a154ff676b9ab70`. Source comparison
confirmed the compatibility described above, and that revision is now accepted.
The updated harness still needs to be run on the server; there are no measured
baseline results yet. All compilation, tests and performance measurements are
performed by the user on the server; local WSL is used for editing and source
review.
