# API

## Common data representation

- E2M1 input/weight: two logical FP4 values per byte, low nibble first;
- UE4M3 scale: raw byte in SM1xx block-scaled physical layout;
- accumulation: FP32;
- output: FP16;
- all device tensors must be contiguous and on the same CUDA device.

E2M1 encodes 1.0 as nibble `0x2`, so a byte containing two 1.0 values is `0x22`. UE4M3 encodes 1.0 as `0x38`.

## GEMM backends

All GEMM paths compute:

```text
C[M,N] = (A_payload * SFA) @ (B_payload * SFB)^T
```

Shapes:

```text
A      [M, K/2] bytes
B      [N, K/2] bytes
SFA    align_up(M,128) * scale_k_padded(K) bytes
SFB    align_up(N,128) * scale_k_padded(K) bytes
C      [M, N] half
```

Constraints: `M > 0`, `K % 32 == 0`, `N % 8 == 0`.

### C++ generic Custom CuTe

The explicit Custom CuTe entry point needs no global workspace:

```cpp
#include <sm120_nvfp4/gemm.hpp>

auto status = sm120_nvfp4::nvfp4_cute_gemm_sm120(
    m, n, k,
    a_packed, b_transposed_packed,
    sfa, sfb, output_fp16, stream);
```

The default entry point is a shape dispatcher:

| M | Selected path | Split-K | Workspace |
|---:|---|---:|---:|
| 128 | M=128 specialization | 4 | `4*M*N*sizeof(float)` |
| 256 | M=256 specialization | 2 | `2*M*N*sizeof(float)` |
| other | generic Custom CuTe | 1 | 0 |

The specialization is selected only when its N/K alignment and K-partition
constraints are satisfied; otherwise the dispatcher safely falls back to the
generic path. Query and allocate the exact workspace before launch:

```cpp
std::size_t bytes =
    sm120_nvfp4::nvfp4_gemm_workspace_size_sm120(m, n, k);
void* workspace = nullptr;
if (bytes != 0) {
  cudaMalloc(&workspace, bytes);
}

auto status = sm120_nvfp4::nvfp4_gemm_sm120(
    m, n, k, a_packed, b_transposed_packed,
    sfa, sfb, output_fp16, workspace, bytes, stream);
```

`nvfp4_gemm_path_sm120(m,n,k)` exposes the deterministic selection for
diagnostics. `nvfp4_cute_gemm_sm120` remains the explicit zero-workspace
generic entry point.

### C++ CUTLASS reference

The previous Collective implementation remains available under an explicit reference name:

```cpp
std::size_t bytes =
    sm120_nvfp4::nvfp4_cutlass_gemm_workspace_size_sm120(m, n, k);

auto status = sm120_nvfp4::nvfp4_cutlass_gemm_sm120(
    m, n, k, a_packed, b_transposed_packed,
    sfa, sfb, output_fp16,
    workspace, bytes, stream);
```

### Python

```python
# Default path automatically specializes M=128/M=256.
y = sm120_nvfp4.gemm(a, b, sfa, sfb)

# Explicit generic path, useful as a stable baseline.
y_cute = sm120_nvfp4.cute_gemm(a, b, sfa, sfb)

# Reference-only path.
y_cutlass = sm120_nvfp4.cutlass_gemm(a, b, sfa, sfb)
```

Registered operators:

```text
torch.ops.sm120_nvfp4.gemm
torch.ops.sm120_nvfp4.cute_gemm
torch.ops.sm120_nvfp4.cutlass_gemm
```

## DS-V4 CSA sparse MLA decode and prefill

The legacy dense prefill and dense/paged decode APIs have been removed.
Use `sparse_mla_decode` with BF16 query/output, shared-latent NVFP4 caches
and explicit SWA/compressed physical slot lists.

`sparse_mla_prefill` takes BF16 `query[T,64,512]`, the same packed cache pools,
and independent int32 `swa_indices[T,Kswa]` / `compressed_indices[T,Kcompressed]`.
Optional arguments are `swa_lengths[T]`, `compressed_lengths[T]`, `sink[64]`,
`softmax_scale`, `lse_scale`, reusable BF16 `output[T,64,512]` and FP32
`lse[T,64]`. It returns `(output, lse)` and has no workspace or split setting.
One CTA processes all candidate chunks for each query. T must be positive;
causality, request isolation and cache lifetime are the caller's responsibility.
Prefill preserves the same precision contract as direct-output decode.

C++ uses `SparseMlaDecodeParams` / `sparse_mla_decode_sm120` and
`SparseMlaPrefillParams` / `sparse_mla_prefill_sm120`. Both parameter types
inherit common fields from `SparseMlaCommonParams`; rebuild callers when
updating the header. The registered prefill op is
`torch.ops.sm120_nvfp4.sparse_mla_prefill`.

See [Sparse MLA](SPARSE_MLA.md) for cache ABI, precision, masking, workspace,
examples and server validation. C++ declarations are in
`sm120_nvfp4/sparse_mla.hpp`.

## C++ Grouped GEMM

The low-level asynchronous entry point is declared in `grouped_gemm.hpp`. It intentionally exposes scratch buffers because it performs no internal allocation or synchronization.

```text
x             [sum_m, K/2]
weight        [groups, N, K/2]
seqlens       [groups] int32
cu_seqlens    [groups + 1] int32
x_scale       [groups, m_scale_pad * scale_k_padded(K)]
weight_scale  [groups, align_up(N,128) * scale_k_padded(K)]
output        [sum_m, N] half
```

`sum(seqlens)` must equal `sum_m`. `m_scale_pad` must be at least the maximum group M and a multiple of 128.

Scratch requirements:

```text
TensorMap bytes  >= (3 * groups + 2) * 128
tiles            >= groups * sizeof(int32)
cu_tiles         >= (groups + 1) * sizeof(int32)
```

## Python Grouped GEMM

```python
y = sm120_nvfp4.grouped_gemm(
    x, weight, seqlens, cu_seqlens, x_scale, weight_scale,
    output=optional_output,
    workspace=optional_tma_workspace,
)
```

The binding allocates tile metadata internally. `workspace` refers only to TensorMap storage and must be a contiguous aligned CUDA `uint8` tensor if supplied.

## Python Fused MoE

```python
y = sm120_nvfp4.fused_moe(
    x,
    x_scale,
    gate_up_weight,
    gate_up_weight_scale,
    down_weight,
    down_weight_scale,
    topk_ids,
    topk_weights,
    shared_output=None,
    ep_rank=0,
    num_experts=None,
    output=None,
)
```

Shapes:

```text
x                       [tokens, hidden/2]
gate_up_weight          [local_experts, 2*intermediate, hidden/2]
down_weight             [local_experts, hidden, intermediate/2]
topk_ids                [tokens, top_k] int32
topk_weights            [tokens, top_k] float32
shared_output/output    [tokens, hidden] float16
```

Constraints:

- hidden and intermediate are multiples of 32;
- local experts are in `[1,256]`;
- top-k is in `[1,128]`;
- local experts occupy `[ep_rank*local_experts, (ep_rank+1)*local_experts)`;
- routes outside the local interval do not contribute to the local result.

## Dispatcher hand-off API

A dispatcher that already produces expert-major expanded rows should avoid the
routing work in `fused_moe`. First quantize its BF16/FP16 receive buffer:

```python
x_nvfp4, x_scale = sm120_nvfp4.quantize_expert(
    recv_x,
    seqlens,
    cu_seqlens,
    scale_m_pad=per_expert_scale_capacity,
    output=optional_packed_buffer,
    output_scale=optional_scale_buffer,
)
```

Then run only the local Expert compute:

```python
workspace_bytes = sm120_nvfp4.expert_moe_workspace_bytes(
    total_rows, hidden, intermediate, local_experts, scale_m_pad
)
workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda")

expert_output = sm120_nvfp4.expert_moe(
    x_nvfp4,
    x_scale,
    gate_up_weight,
    gate_up_weight_scale,
    down_weight,
    down_weight_scale,
    seqlens,
    cu_seqlens,
    output=optional_fp16_output,
    workspace=workspace,
)
```

`seqlens` and `cu_seqlens` describe the already grouped rows;
`sum(seqlens) == recv_x.size(0)`. `scale_m_pad` is a multiple of 128 and
must cover the largest Expert M. The output remains in expanded expert-major
order, one row per route; top-k weighting and cross-rank reduction belong to
the dispatcher's Combine step.

The workspace stores gate/up output, dynamically quantized SwiGLU output and
scales, both GEMMs' TensorMaps, and tile-prefix metadata. Reusing it avoids
per-call temporary tensor allocation. It is stream-ordered and must stay alive
until the current CUDA stream completes.

## Synchronization

All entry points launch on the caller-provided/current CUDA stream and do not synchronize. Output and workspace tensors must remain alive until the stream completes.
