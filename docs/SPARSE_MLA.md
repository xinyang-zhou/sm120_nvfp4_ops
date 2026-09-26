# DS-V4 CSA sparse MLA decode and prefill on SM120

Status: implementation prepared in WSL; **not compiled, tested or benchmarked
locally**. Server validation is required. The former dense prefill and
dense/paged decode APIs, kernels and tests have been removed. Sparse MLA is
the repository's attention implementation.

The numerical design is recorded in [attention_decode_math.tex](attention_decode_math.tex),
a snapshot of `/home/xinyang/attention.tex`. This implements the attention core:
the caller supplies post-RoPE Q, packed shared-latent KV pools and selected
physical slot IDs. Compression, indexer/Top-K selection, cache append, causal
selection and output projection remain outside this operator.

## Interface and precision

```python
import torch
import sm120_nvfp4

# q: BF16 [B,64,512] or [B,1,64,512]
# Each cache: uint8 [pages,64,384], or singleton-head HND/NHD equivalent.
# Each index list: CUDA int32 [B,K] or [B,1,K]. K may be zero or nonaligned.
# sink: optional FP32 [64], natural logits; lengths: optional int32 [B].
workspace = torch.empty(
    sm120_nvfp4.sparse_mla_decode_workspace_bytes(B, Kswa, Kcompressed, 0),
    device=q.device, dtype=torch.uint8,
)
out, lse2 = sm120_nvfp4.sparse_mla_decode(
    q, swa_cache, compressed_cache, swa_indices, compressed_indices,
    swa_lengths=swa_lengths, compressed_lengths=compressed_lengths,
    sink=sink, chunks_per_cta=0, workspace=workspace,
)
```

Output has Q's shape/dtype. LSE is FP32 `[B,64]`, base-2 by default; set
`lse_scale=math.log(2)` for natural logarithms. Reusable `output`, `lse`, and
`workspace` buffers avoid allocation in the binding. Buffers must not overlap.
Inputs must be contiguous on the same GPU. No autograd support.

### Fixed-index sparse prefill

`sparse_mla_prefill` accepts packed query rows from one or more requests:

```python
# q: contiguous BF16 [T,64,512], T=sum(query_lengths), 1 <= T <= 1048576.
# Both caches are shared across query rows, with the same ABI as decode.
# Each query has its OWN list: int32 [T,Kswa] and [T,Kcompressed].
# Optional valid lengths are int32 [T]; sink is optional FP32 [64].
out, lse2 = sm120_nvfp4.sparse_mla_prefill(
    q, swa_cache, compressed_cache, swa_indices, compressed_indices,
    swa_lengths=swa_lengths, compressed_lengths=compressed_lengths,
    sink=sink,
    output=reusable_output,  # optional BF16 [T,64,512]
    lse=reusable_lse,        # optional FP32 [T,64]
)
```

The prefill interface requires 3D Q/output and 2D indices. An equal-length
batch `[B,L,64,512]` can be flattened to `[B*L,64,512]`, with indices and
lengths flattened in the same order. Ragged requests are concatenated along
T; request boundaries and absolute positions stay with the caller because
all selection is already encoded in physical slot lists. Candidate capacities
are shared across rows in a call, independently in `[0,1048576]` for each pool;
shorter lists use lengths and/or negative padding. An empty request contributes
no rows; skip the call when the total T is zero.

One CTA processes all candidate chunks for one query, even when T<=64. There
is one attention kernel launch and no split/merge, temporary global workspace,
or per-query host loop. With reusable output/LSE, the binding allocates no
device tensors. The math is the decode direct-output path: NVFP4 non-RoPE,
BF16 RoPE and W rounding, FP32 unquantized denominator and output accumulators,
candidate-axis V requantization, and a single final sink/normalization step.

For query-chunked calls, slice Q, both index tensors and both length tensors
by the same query interval, keeping the cache pools, sink, candidate order
and capacities unchanged. Each query retains its 64-candidate chunk boundaries
and 16-candidate quantization groups. Results should match a whole-query-batch
call exactly on the same build/device; the server tests enforce this. This
does not imply invariance to reordering candidates or changing the split
schedule: quantization and BF16 split rounding can change those results.

The caller must supply causally valid SWA and compressed indices per query,
ensure request isolation, and retain referenced KV rows until the GPU finishes.
In particular, prefill cannot read history from SWA ring slots that have already
been overwritten. This core does not check logical positions, generate Top-K,
compress/append KV, or maintain state across chunks. The chunking tests cover
fixed-index attention output consistency, not compressor/cache state evolution.

The C++ entry point is `sparse_mla_prefill_sm120(SparseMlaPrefillParams, stream)`
in `sm120_nvfp4/sparse_mla.hpp`. Set `num_queries=T` and the common pointers,
page counts, capacities and scales. It takes no workspace and asynchronously
launches on the supplied stream. Decode retains its `batch`/`chunks_per_cta`
fields; both parameter types share `SparseMlaCommonParams`. Rebuild C++ callers
against the updated header rather than reusing an old binary parameter layout.

| Component | Precision and operation |
|---|---|
| Query | BF16; first 448 channels quantized online in groups of 16 |
| Cache non-RoPE | E2M1 payload, E4M3 scale per 16 channels |
| Cache RoPE | Last 64 channels remain BF16 |
| QK | NVFP4 `m16n8k64` then BF16 `m16n8k16`, FP32 accumulation |
| Softmax | FP32 local/CTA max and exponent sum; no quantized-denominator correction |
| PV non-RoPE | BF16-rounded W -> NVFP4 P; decoded V requantized per 16 candidates |
| PV RoPE | BF16 W and BF16 V, FP32 accumulation |
| Across chunks | FP32 maximum/denominator in shared memory; FP32 output in registers |
| Split scratch | Normalized BF16 output and FP32 base-2 LSE |
| Sink/final output | Sink added once; FP32 merge and one final BF16 conversion |

Scale rounding to zero produces zero payload values, with no minimum-scale
fallback. Cache payloads are shared between K and V, but their quantization
axes differ, so V conversion is done inside the CTA. This is a mixed-precision
NVFP4 experiment matching the documented native path, not a claim of original
model precision equivalence or bitwise equivalence to FlashInfer.

## Cache format and masking

A page is `64*384` bytes. The first `64*352` bytes hold 64 data rows. Each data
row has 224 packed FP4 bytes followed by 128 BF16 RoPE bytes. The remaining
`64*32` bytes hold the scale footer: 28 E4M3 bytes plus four padding bytes per
token. **Do not interpret the tensor's apparent 384-byte rows as token records.**
The format accepts FlashInfer's `nvfp4_quantize_pack_sparse_mla_cache` output.

An index `i` selects physical page `i//64`, row `i%64` in its own pool. SWA
chunks come before compressed-cache chunks; each list is padded independently
to 64 candidates. Negative/out-of-range indices, list padding, and positions
past the optional per-query length contribute neither numerator nor
denominator. Lengths are clamped to the list capacity. Duplicate IDs count
once per occurrence. The caller decides causal visibility and SWA contents.

Both lists and pools may be empty. Empty split outputs are zero with the
internal LSE sentinel `-1e30`. Queries with no valid candidates return zero output, with
sink-only LSE or `-inf` without a finite sink. Q/cache values must be finite;
sink values may be finite or `-inf`.

## Implementation and scheduling

`src/attention/sparse_mla/mma.cuh` uses C++ CuTe `MMA_Atom`, its A/B/C/scale
layouts, tensors and `cute::gemm`. It does not call FlashInfer or a CUTLASS
device GEMM adapter. CuTe selects the native SM120 NVFP4 instruction; the BF16
atom is the warp-level instruction also available on SM120.

`decode.cu` defines the shared `attention_kernel`, with eight warps per CTA.
`prefill.cu` selects its all-chunk, direct-output schedule. Each CTA handles all 64 heads and
one contiguous range of 64-candidate chunks. It quantizes Q once, gathers
raw KV, computes QK, updates softmax, prepares P/V, and accumulates both PV
branches. Source W and destination P have separate buffers. The raw KV uses
two buffers with `cp.async` prefetch of the next chunk and explicit wait/CTA
barriers before consumption or reuse. Shared memory is 95 KiB per CTA.

Compared with the reference kernel described in the tex, this version uses
the eight compute warps for V conversion after QK rather than separate IO
warps converting V concurrently. It loads Q's BF16 RoPE part from global
memory during QK. The merge uses one CTA/head. These are scheduling choices;
the rounding/normalization/sink contract is preserved. Performance, register
spills and occupancy must be measured on the server.

For decode, `chunks_per_cta=0` uses up to nine chunks/CTA for B<=64 and all chunks for
larger batches. For 128+512 candidates this gives two splits at B=64 and one
at B=256/1024. This reproduces the initial documented plan, not an autotuned
claim. Positive values override it. Workspace is zero for one split; otherwise
it is `B*64*splits*(512*2+4)` bytes. No global per-chunk output/logit/P/V
workspace is used.

Prefill always uses one CTA/query and zero scratch, including T=1 and empty
candidate lists. It keeps the same 95 KiB shared storage and CuTe MMA loop as
decode. Cross-query KV reuse, query tiling and prefill autotuning are future
optimizations; no prefill throughput claim has been measured.

## Server validation

Run from the repository root after pulling the commit. CUDA 12.8+,
CUTLASS 4.2+ and an SM120 GPU are required. Set `CUDA_ROOT`/`CUTLASS_ROOT`
as for the existing GEMM build. A newer toolchain with NVFP4 support is
recommended if the installed CUDA headers do not provide `cuda_fp4.h`.

```bash
git fetch origin
git switch main
git pull --ff-only
mkdir -p artifacts/sparse_mla
set -o pipefail
git rev-parse HEAD | tee artifacts/sparse_mla/commit.txt
nvidia-smi > artifacts/sparse_mla/nvidia-smi.txt
bash scripts/build.sh 2>&1 | tee artifacts/sparse_mla/build.log
export PYTHONPATH="$PWD/build/python${PYTHONPATH:+:$PYTHONPATH}"
python3 -c 'import torch; assert torch.cuda.get_device_capability() == (12, 0)'
python3 -m unittest discover -s tests/python -p 'test_sparse_mla*.py' -v \
  2>&1 | tee artifacts/sparse_mla/sparse_mla.log
```

Required gates: build succeeds; all non-optional sparse tests pass on SM120
(not skip). The sparse suite checks all 64 heads, random
per-block scales, tails, masked/repeated/out-of-range indices, lengths, empty
pools/splits, zero scales, denominator precision, BF16 RoPE, large logits,
sink, split rounding, buffer reuse, API validation and CUDA Graph replay.
Elementwise output and RMSE are checked against the independent matrix
reference; LSE has a tighter FP32 tolerance. These gates do not establish model
quality on real weights/activations.

The prefill suite additionally checks T=1/7/65, full 128+512 candidates,
per-query lengths, ragged request rows with disjoint slot ranges, nonaligned
query chunks, direct-decode equivalence, ignored cache-row mutation, empty
lists, the FP32-denominator/BF16-RoPE contract, shape/alias rejection and CUDA
Graph replay after input mutation. Its cache/selection fixtures are synthetic;
they do not implement a real CSA compressor or indexer.

Use a reviewed FlashInfer checkout (`37b4d30eac39b89f198b893dd11914bd76f5fcf8`
or `ea728cb558c32a3c58ec8fbd5a154ff676b9ab70`) for the integration test:

```bash
SPARSE_MLA_FLASHINFER_TEST=1 python3 -m unittest discover \
  -s tests/python -p 'test_sparse_mla*.py' -v \
  2>&1 | tee artifacts/sparse_mla/flashinfer.log
compute-sanitizer --tool memcheck --error-exitcode 1 \
  python3 tests/python/test_sparse_mla.py SparseMlaTest.test_tail_masks_repeated_ids_and_lengths \
  2>&1 | tee artifacts/sparse_mla/memcheck.log
compute-sanitizer --tool racecheck --error-exitcode 1 \
  python3 tests/python/test_sparse_mla.py SparseMlaTest.test_csa_128_plus_512_and_split_rounding \
  2>&1 | tee artifacts/sparse_mla/racecheck.log
compute-sanitizer --tool memcheck --error-exitcode 1 \
  python3 tests/python/test_sparse_mla_prefill.py SparseMlaPrefillTest.test_per_query_lengths_tails_and_masks \
  2>&1 | tee artifacts/sparse_mla/prefill_memcheck.log
compute-sanitizer --tool racecheck --error-exitcode 1 \
  python3 tests/python/test_sparse_mla_prefill.py SparseMlaPrefillTest.test_csa_full_lists_and_direct_decode \
  2>&1 | tee artifacts/sparse_mla/prefill_racecheck.log
python3 benchmarks/benchmark_sparse_mla.py --batches 64 256 1024 \
  --rounds 3 --samples 100 --flashinfer \
  --output artifacts/sparse_mla/benchmark.json \
  2>&1 | tee artifacts/sparse_mla/benchmark.log
```

The integration tests compare cache packing and decode outputs/LSE with fixed
CPB, and prefill outputs/LSE with FlashInfer's streaming prefill entry point.
Sanitizers must report zero errors. Benchmark validates sampled requests
before timing and records all samples/P50/P95, seed, commits, GPU, buffers and
peak allocation. The existing benchmark measures decode; prefill performance
has no recorded result. The public FlashInfer decode baseline chooses its own schedule;
record its actual kernel/CPB with a profiler before claiming a controlled
schedule comparison. Both calls use identical inputs, preallocated output and
scratch, and the same graph/eager and L2-flush settings. Peak allocation also
includes construction/reference, not just kernel scratch.

Keep complete build logs (including compiler diagnostics), test assertions,
sanitizer reports and JSON if any step fails. No server result is recorded as
passed in this commit.
