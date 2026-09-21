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

## Dense NVFP4 prefill attention

Both matrix products use the repository-owned SM120 block-scaled MMA:

```text
logits = Q_nvfp4 @ K_nvfp4^T                  (FP32 output)
P       = softmax(logits * softmax_scale)     (FP32)
P_nvfp4 = dynamic_quantize(P, groups_of_16)   (E2M1 + UE4M3)
output  = P_nvfp4 @ V_transposed_nvfp4^T      (FP16)
output *= 1 / sum(dequantize(P_nvfp4), axis=N)
```

The final correction restores the probability-row normalization lost during
E2M1 quantization. It does not remove relative per-element quantization error.

Packed shapes and scale-region sizes per `(batch, head)` are:

```text
query              [B, H, M,  D/2]
key                [B, H, N,  D/2]
value_transposed   [B, H, Dv, N/2]
query_scale        B*H * scale_a_elements(M, N,  D)
key_scale          B*H * scale_b_elements(M, N,  D)
value_scale        B*H * scale_b_elements(M, Dv, N)
output             [B, H, M, Dv] float16
```

`value_transposed` stores rows of `V^T`, because the SM120 GEMM contract is
`A[M,K] @ B[N,K]^T`. Constraints are `D % 32 == 0`, `N % 32 == 0`, and
`Dv % 8 == 0`. With `causal=True`, `M <= N` and row `r` can see KV columns
through `N-M+r`, matching suffix-aligned prefill semantics.

```python
workspace = torch.empty(
    sm120_nvfp4.attention_workspace_bytes(B, H, M, N),
    dtype=torch.uint8,
    device=query.device,
)
output = sm120_nvfp4.attention_prefill(
    query, key, value_transposed,
    query_scale, key_scale, value_scale,
    causal=True,
    softmax_scale=D**-0.5,  # default when omitted
    workspace=workspace,
)
```

The C++ declarations are in `sm120_nvfp4/attention.hpp`. The workspace holds
FP32 logits, packed probability payloads, probability scales and FP32 row
corrections; the query function returns the exact required byte count.

This implementation materializes logits and probabilities and launches GEMMs
per `(B,H)` matrix. An end-to-end persistent/fused schedule is not yet part of
the API.

## Dense NVFP4 decode attention

The decode entry point handles one query token per request and supports GQA
and MQA without expanding the KV cache:

```text
query                    [B, Hq, D/2]
key_cache                [B, Hkv, N, D/2]
value_cache_transposed   [B, Hkv, Dv, N/2]
kv_lengths               [B] int32 CUDA, optional
output                   [B, Hq, Dv] float16
```

`Hq % Hkv == 0`. Consecutive groups of `Hq/Hkv` query heads directly reuse
one KV head's packed payload and scales. `N` is padded cache capacity;
`kv_lengths[b]` masks positions at or beyond the request's current length.
When omitted, all `N` positions are valid.

Scale sizes are:

```text
query_scale  B*Hq  * scale_a_elements(1, N,  D)
key_scale    B*Hkv * scale_b_elements(1, N,  D)
value_scale  B*Hkv * scale_b_elements(1, Dv, N)
```

```python
workspace = torch.empty(
    sm120_nvfp4.attention_decode_workspace_bytes(B, Hq, Hkv, N, D, Dv),
    dtype=torch.uint8,
    device=query.device,
)
output = sm120_nvfp4.attention_decode(
    query, key_cache, value_cache_transposed,
    query_scale, key_scale, value_scale,
    kv_lengths=kv_lengths,
    softmax_scale=D**-0.5,  # default when omitted
    workspace=workspace,
)
```

The decode kernel groups the `Hq/Hkv` query heads sharing one KV head into the
M dimension and streams over 128-token KV tiles. Each tile performs native
SM120 NVFP4 QK, online FP32 softmax, tile-local E2M1/UE4M3 probability
quantization, and native NVFP4 PV before the tile storage is reused. No full
`[B,Hq,N]` logits or probability tensor is written to global memory.

For small request counts, the sequence is split across CTAs. Workspace holds
only grouped query scales, FP32 `[B,Hq,splits,Dv]` partial outputs, and
`[B,Hq,splits]` LSE values; a second kernel combines them stably. Packed K/V
payloads remain unexpanded. Paged block tables and multi-token prediction are
not part of this dense API.

## Paged NVFP4 decode attention

`attention_paged_decode` keeps the same single-token GQA/MQA and split-K
semantics, but resolves logical tokens through a device block table:

```text
query                    [B, Hq, D/2]
key_cache                [P, Hkv, S, D/2]
value_cache_transposed   [P, Hkv, Dv, S/2]
block_table              [B, max_blocks] int32 CUDA
kv_lengths               [B] int32 CUDA
output                   [B, Hq, Dv] float16
```

`S` is 32, 64, or 128 and `N = max_blocks * S`. Logical token `t` of
request `b` comes from physical page `block_table[b,t//S]` at offset `t%S`.
Entries covering the active `ceil(kv_lengths[b]/S)` pages must be in `[0,P)`.
Physical pages may be shared or appear in any order.

Each `(physical page, KV head)` owns an independent CUTLASS SFB region:

```text
query_scale  B*Hq  * scale_a_elements(1, N,  D)
key_scale    P*Hkv * scale_b_elements(1, S,  D)
value_scale  P*Hkv * scale_b_elements(1, Dv, S)
```

```python
workspace = torch.empty(
    sm120_nvfp4.attention_decode_workspace_bytes(B, Hq, Hkv, N, D, Dv),
    dtype=torch.uint8,
    device=query.device,
)
output = sm120_nvfp4.attention_paged_decode(
    query, key_cache, value_cache_transposed,
    query_scale, key_scale, value_scale,
    block_table, kv_lengths,
    softmax_scale=D**-0.5,
    workspace=workspace,
)
```

The CTA reads complete packed E2M1 byte pairs through the block table and
places them directly in the existing swizzled QK/PV shared-memory tiles.
This avoids expanding the cache to `[B,Hkv,N,*]`; online FP32 softmax,
tile-local probability quantization, native SM120 NVFP4 QK/PV, and LSE
combine are shared with dense decode.

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
