# Architecture

## Layered design

The project is organized as one operator stack rather than three unrelated examples.

```text
Layer 1: Single GEMM
  shape dispatcher ─┬─ M=128 Split-K=4 ─┐
                    ├─ M=256 Split-K=2 ─┼─ repository-owned kernels
                    └─ generic CuTe ─────┤
  CUTLASS Collective reference ──────────┼─ correctness/performance comparison
  cuBLASLt baseline ─────────────────────┘
                    |
          ┌─────────┴──────────┐
          v                    v
Layer 2A: Dense Attention   Layer 2B: Persistent Grouped GEMM
  QK^T -> softmax -> P@V      dynamic expert M + shared N/K
                               |
                               v
Layer 3: Fused MoE
  route/count/gather
        -> grouped gate/up GEMM
        -> SiLU(gate) * up + dynamic NVFP4 quantization
        -> grouped down GEMM
        -> weighted top-k reduce
```

## Custom CuTe GEMM

All single-GEMM paths compute:

```text
C[M,N] = (A_e2m1[M,K] * SFA) @ (B_e2m1[N,K] * SFB)^T
```

[`src/gemm/cute_gemm.cu`](../src/gemm/cute_gemm.cu) owns and launches the
generic CUDA kernel. It does not instantiate
`cutlass::gemm::kernel::GemmUniversal` or
`cutlass::gemm::device::GemmUniversalAdapter`.

The current specialization uses:

- CTA tile `128 x 128 x 128`;
- 384 threads: 256 MMA consumers and 128 producer threads;
- three shared-memory stages;
- TMA loads for A, B, SFA and SFB;
- one persistent grid capped by the number of resident SMs;
- direct `cute::partition_*`, `cute::copy` and `cute::gemm` calls;
- a hybrid FP16 epilogue: shared-memory staging plus TMA store for
  `M >= 64`, and predicated direct stores for smaller M.

Each CTA advances through flattened `(tile_m,tile_n)` work in a grid-stride loop. The producer warp owns TMA issue and transaction barriers. The 256 math threads copy the packed payload/scales from shared memory, apply FP4 shifts, issue block-scaled MMA and publish each stage through consumer barriers.

For the TMA epilogue, the math threads convert their FP32 accumulators into a
dense row-major FP16 shared-memory tile. After a named-barrier and async-shared
fence, one elected thread issues `SM90_TMA_STORE`; the producer warpgroup may
continue filling the disjoint mainloop buffers. TensorMap bounds discard
residue outside M/N. A second named-barrier prevents reuse of the single
epilogue tile before the store completes. FP32 output, used by prefill logits,
retains the direct predicated path because a full FP32 staging tile would
exceed the useful shared-memory budget.

### Shape dispatcher and fixed-M paths

[`src/gemm/gemm_dispatch.cu`](../src/gemm/gemm_dispatch.cu) is the only
default-path selector. It routes supported `M=128` shapes to Split-K=4,
supported `M=256` shapes to Split-K=2, and all other shapes to the generic
kernel. Selection depends only on `(M,N,K)` and can be inspected through
`nvfp4_gemm_path_sm120`.

The two implementations live under
[`src/gemm/specialized/`](../src/gemm/specialized). They produce FP32 partial
matrices in caller-owned workspace and use a second kernel to reduce and
convert to FP16. The split counts are fixed, not runtime tuning parameters.
The M=256 path additionally fixes its tile-to-task mapping and uses a
vectorized `half2` reduction. Attention continues to call the explicit
generic FP16/FP32 and batched interfaces; it is not redirected through this
single-GEMM dispatcher.

`cuobjdump` confirms that this repository-owned kernel contains:

```text
OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X
```

## What “Custom CuTe” means here

CuTe is distributed inside CUTLASS, and SM120 support depends on CUTLASS headers for element types, barriers, TMA atoms and the architecture-specific block-scaled MMA selector. The shared [`gemm_config.cuh`](../src/common/gemm_config.cuh) also derives validated shared-memory layouts and copy atoms from the SM120 CollectiveMainloop type.

The ownership boundary is therefore:

| Component | Owner |
|---|---|
| CTA scheduling and persistent traversal | this repository |
| producer/consumer split and barrier protocol | this repository |
| TMA issue order and stage rotation | this repository |
| register fragments, `cute::gemm`, predication and epilogue | this repository |
| CuTe tensor/layout primitives and SM120 MMA atoms | CUTLASS/CuTe dependency |
| `GemmUniversal` device wrapper | not used by Custom CuTe path |

The separate [`cutlass_reference.cu`](../src/gemm/cutlass_reference.cu) deliberately uses `CollectiveBuilder + GemmUniversalAdapter`; it is retained only as a reference.

## Shared single/grouped configuration

Single and Grouped GEMM share [`Nvfp4GemmConfig`](../src/common/gemm_config.cuh). This guarantees that both paths use the same:

- E2M1/UE4M3 types;
- `rr_blockscaled_op_selector_sm120` MMA atom;
- `TiledMma` shape;
- shared-memory layouts and copy atoms;
- TMA copy types and transaction byte count.

The single kernel specializes scheduling for one matrix. Grouped GEMM adds runtime TensorMap patching and expert-aware tile traversal without changing the MMA contract.

## Dense attention

### Prefill

The first SM120 attention path borrows the numerical decomposition used by
high-performance prefill kernels while replacing SM90 WGMMA with this
repository's SM120 block-scaled MMA:

1. each `(batch, head)` computes packed NVFP4 `QK^T` with FP32 output;
2. one CUDA block per query row applies suffix-aligned causal masking and a
   stable FP32 max/sum softmax;
3. each 16-element probability vector is dynamically quantized to E2M1 with
   a UE4M3 scale in the CUTLASS SFA physical layout;
4. each `(batch, head)` computes NVFP4 `P@V` into FP16;
5. a final row correction divides by the reconstructed NVFP4 probability sum.

This preserves the attention normalization invariant while letting both
matrix products use `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`. The current path
is a correctness-oriented first implementation: FP32 logits and packed
probabilities live in caller-reusable workspace, and GEMMs are launched per
head. A future fused schedule can tile QK, online softmax and PV without
changing the public operand/scale contract.

### Decode

Single-token dense decode uses a fused 128-token streaming schedule. The
`Hq/Hkv` query heads sharing one cache head become the MMA M dimension. Within
each CTA, TMA feeds native SM120 NVFP4 QK, the resulting FP32 tile is consumed
by online softmax, probabilities are quantized into shared-memory E2M1/UE4M3,
and a second native NVFP4 MMA accumulates PV. The same 64 KiB shared-memory
region is reused for Q/K storage and the temporary logits tile, so neither
full logits nor full probabilities reach global memory.

Low-occupancy requests split the KV sequence across CTAs and use a stable
LSE-weighted combine. Each split writes only an output-sized FP32 partial and one
LSE per query head; a small combine kernel applies stable LSE weights. GQA/MQA
reuses packed K/V without expansion. Per-head query scales are repacked into
grouped tiles in reusable workspace. Device `kv_lengths[B]` supplies the last
valid position without host synchronization.

The SM120 implementation organizes online softmax and split-K/LSE around
block-scaled `OMMA.SF` for both QK and PV; it does not reuse an SM90 WGMMA
schedule.

The paged entry point adds block-table indirection without
materializing a dense cache. Physical caches are `[P,Hkv,S,D]` for K and
`[P,Hkv,Dv,S]` for transposed V, with `S` equal to 32, 64, or 128. A CTA maps
each logical 128-token tile to its physical pages, loads complete packed E2M1
byte pairs into the same swizzled B-operand shared layout, and loads the
corresponding per-page SFB metadata. Handling both nibbles in one thread is
required: independent subbyte stores would race on a shared packed byte.
After that load, paged and dense decode share the same native QK, online
softmax, probability quantization, native PV, and LSE combine code. MTP and a
dynamic device task map remain the next decode extensions.

## Grouped GEMM

Grouped GEMM accepts concatenated expert activations and one weight matrix per expert. Each expert has a dynamic M described by `seqlens` and `cu_seqlens`; N and K are shared.

Before the persistent compute launch, small CUDA kernels:

1. patch per-expert activation TensorMaps;
2. patch per-expert SFA TensorMaps;
3. publish static weight/SFB TensorMaps;
4. compute per-expert tile counts and prefix sums.

One persistent grid then consumes tiles across experts. This removes one host GEMM launch per expert and is most valuable when each expert receives few rows.

## Fused MoE

Fused MoE reuses Grouped GEMM twice. It is not presented as one monolithic CUDA kernel; one public operator owns and stream-orders the full local-expert pipeline without exposing intermediates to the caller.

1. count local routes and build expert-major offsets;
2. gather packed input and SFA blocks;
3. gate/up Grouped GEMM;
4. compute `SiLU(gate) * up`, quantize E2M1 and produce UE4M3 scales;
5. down Grouped GEMM;
6. scatter/reduce by top-k weights, optionally adding shared-expert output.

## Dispatcher hand-off

DeepEP-style expand dispatch already performs the first two local routing
steps: it returns rows grouped by local Expert, per-Expert counts/prefix
offsets, route weights and a handle for reverse Combine. Feeding that output
back into the full `fused_moe` path would count and gather the same routes a
second time.

The `expert_moe` entry point therefore splits the ownership boundary at the
dispatcher/computation interface:

```text
Expand Dispatch
  -> expert-major BF16 rows + seqlens/cu_seqlens + route handle
  -> per-Expert BF16-to-NVFP4 quantization
  -> grouped gate/up GEMM
  -> SiLU(gate) * up + dynamic NVFP4 quantization
  -> grouped down GEMM
  -> expanded FP16 rows
  -> dispatcher Combine
```

This is a direct layout hand-off, not a literal zero-copy path: BF16/FP8
communication output still needs conversion to the packed E2M1 payload and
UE4M3 scale layout consumed by SM120 Tensor Cores. The persistent workspace
keeps the conversion output, intermediate activations, TensorMaps and tile
metadata at stable addresses across calls.

The checked-in benchmark uses a readable `torch.distributed` reference
transport to validate this contract on two GPUs. It deliberately does not
claim to benchmark DeepEP's native NVLink/RDMA kernels.

## Scale layout

Logical scales are not a flat `[M,K/16]` matrix. They use the physical layout produced by `cutlass::detail::Sm1xxBlockScaledConfig<16>`.

```text
scale_k_padded(K) = align_up(ceil(K / 16), 4)
SFA bytes          = align_up(M, 128) * scale_k_padded(K)
SFB bytes          = align_up(N, 128) * scale_k_padded(K)
```

Uniform tests can fill the physical allocation with one UE4M3 byte. Non-uniform application data must be packed into the physical layout.

## Ownership boundaries

- `include/`: supported C++ entry points and contracts;
- `src/common/`: shared CuTe configuration and device helpers;
- `src/gemm/`: default dispatcher, generic/specialized Custom CuTe kernels,
  and isolated CUTLASS reference;
- `src/grouped_gemm/`: dynamic expert scheduling and compute kernel;
- `src/attention/`: materialized prefill plus fused streaming/split-K decode;
- `bindings/`: validation, allocation and PyTorch registration;
- `tests/`: correctness and adversarial routing/layout cases;
- `benchmarks/`: performance comparisons, never imported by the library.
