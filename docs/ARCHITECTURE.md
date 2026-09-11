# Architecture

## Layered design

The project is organized as one operator stack rather than three unrelated examples.

```text
Layer 1: Single GEMM
  Custom CuTe kernel ───────────────┐
  CUTLASS Collective reference ─────┼─ correctness/performance comparison
  cuBLASLt baseline ────────────────┘
                    |
                    v
Layer 2: Persistent Grouped GEMM
  dynamic expert M + shared N/K + per-expert TensorMaps
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

The default GEMM computes:

```text
C[M,N] = (A_e2m1[M,K] * SFA) @ (B_e2m1[N,K] * SFB)^T
```

[`src/gemm/cute_gemm.cu`](../src/gemm/cute_gemm.cu) owns and launches the CUDA kernel. It does not instantiate `cutlass::gemm::kernel::GemmUniversal` or `cutlass::gemm::device::GemmUniversalAdapter`.

The current specialization uses:

- CTA tile `128 x 128 x 128`;
- 384 threads: 256 MMA consumers and 128 producer threads;
- three shared-memory stages;
- TMA loads for A, B, SFA and SFB;
- one persistent grid capped by the number of resident SMs;
- direct `cute::partition_*`, `cute::copy` and `cute::gemm` calls;
- predicated FP32-accumulator to row-major FP16 stores.

Each CTA advances through flattened `(tile_m,tile_n)` work in a grid-stride loop. The producer warp owns TMA issue and transaction barriers. The 256 math threads copy the packed payload/scales from shared memory, apply FP4 shifts, issue block-scaled MMA and publish each stage through consumer barriers.

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
- `src/gemm/`: Custom CuTe kernel and isolated CUTLASS reference;
- `src/grouped_gemm/`: dynamic expert scheduling and compute kernel;
- `bindings/`: validation, allocation and PyTorch registration;
- `tests/`: correctness and adversarial routing/layout cases;
- `benchmarks/`: performance comparisons, never imported by the library.
