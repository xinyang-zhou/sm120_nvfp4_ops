"""PyTorch interface for the SM120 NVFP4 operator stack."""

from __future__ import annotations
from pathlib import Path

from typing import Optional

import torch

torch.ops.load_library(str(Path(__file__).with_name("_C.so")))


def gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
) -> torch.Tensor:
    """Compute ``A[M,K] @ B[N,K].T`` from packed E2M1 operands."""
    return torch.ops.sm120_nvfp4.gemm(a, b, sfa, sfb)


def cute_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
) -> torch.Tensor:
    """Run the repository-owned persistent CuTe SM120 kernel."""
    return torch.ops.sm120_nvfp4.cute_gemm(a, b, sfa, sfb)


def cutlass_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
) -> torch.Tensor:
    """Run the CUTLASS Collective reference implementation."""
    return torch.ops.sm120_nvfp4.cutlass_gemm(a, b, sfa, sfb)


def attention_workspace_bytes(
    batch: int,
    heads: int,
    query_length: int,
    kv_length: int,
) -> int:
    """Return reusable scratch bytes for :func:`attention_prefill`."""
    if min(batch, heads, query_length, kv_length) <= 0:
        raise ValueError("attention dimensions must be positive")
    if kv_length % 32:
        raise ValueError("kv_length must be a multiple of 32")

    alignment = 256
    matrices = batch * heads
    total = 0

    def reserve(size: int) -> None:
        nonlocal total
        total = ((total + alignment - 1) // alignment) * alignment
        total += size

    reserve(matrices * query_length * kv_length * 4)
    reserve(matrices * query_length * kv_length // 2)
    reserve(matrices * scale_a_elements(query_length, 1, kv_length))
    reserve(matrices * query_length * 4)
    return ((total + alignment - 1) // alignment) * alignment


def attention_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value_transposed: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    *,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run dense SM120 NVFP4 prefill attention.

    Packed shapes are ``query[B,H,M,D/2]``, ``key[B,H,N,D/2]`` and
    ``value_transposed[B,H,Dv,N/2]``. Softmax is FP32; both matrix products
    use native SM120 NVFP4 block-scaled MMA.
    """
    if softmax_scale is None:
        logical_head_dim = int(query.shape[-1]) * 2
        softmax_scale = logical_head_dim**-0.5
    return torch.ops.sm120_nvfp4.attention_prefill(
        query,
        key,
        value_transposed,
        query_scale,
        key_scale,
        value_scale,
        causal,
        softmax_scale,
        output,
        workspace,
    )


def attention_decode_workspace_bytes(
    batch: int,
    query_heads: int,
    kv_heads: int,
    max_kv_length: int,
    head_dim: int,
    value_dim: int,
) -> int:
    """Return reusable scratch bytes for :func:`attention_decode`."""
    if min(batch, query_heads, kv_heads, max_kv_length, head_dim, value_dim) <= 0:
        raise ValueError("decode dimensions must be positive")
    if query_heads % kv_heads:
        raise ValueError("query_heads must be divisible by kv_heads")
    if max_kv_length % 32 or head_dim % 32:
        raise ValueError("max_kv_length and head_dim must be multiples of 32")
    if value_dim % 8:
        raise ValueError("value_dim must be a multiple of 8")

    alignment = 256
    matrices = batch * query_heads
    groups = batch * kv_heads
    rows_per_group = query_heads // kv_heads
    row_tiles = (rows_per_group + 127) // 128
    sequence_tiles = (max_kv_length + 127) // 128
    tasks = groups * row_tiles
    splits = min(sequence_tiles, max(1, (128 + tasks - 1) // tasks))
    total = 0

    def reserve(size: int) -> None:
        nonlocal total
        total = ((total + alignment - 1) // alignment) * alignment
        total += size

    reserve(groups * scale_a_elements(rows_per_group, max_kv_length, head_dim))
    reserve(matrices * splits * value_dim * 4)
    if splits > 1:
        reserve(matrices * splits * 4)
    return ((total + alignment - 1) // alignment) * alignment


def attention_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache_transposed: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    *,
    kv_lengths: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run single-token dense SM120 NVFP4 decode attention with GQA/MQA.

    Packed shapes are ``query[B,Hq,D/2]``, ``key_cache[B,Hkv,N,D/2]``
    and ``value_cache_transposed[B,Hkv,Dv,N/2]``. ``kv_lengths`` is an
    optional CUDA int32 tensor of shape ``[B]``.
    """
    if softmax_scale is None:
        logical_head_dim = int(query.shape[-1]) * 2
        softmax_scale = logical_head_dim**-0.5
    return torch.ops.sm120_nvfp4.attention_decode(
        query,
        key_cache,
        value_cache_transposed,
        query_scale,
        key_scale,
        value_scale,
        kv_lengths,
        softmax_scale,
        output,
        workspace,
    )


def attention_paged_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache_transposed: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    block_table: torch.Tensor,
    kv_lengths: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run fused single-token paged SM120 NVFP4 decode attention.

    Packed caches are block-major: ``key_cache[P,Hkv,S,D/2]`` and
    ``value_cache_transposed[P,Hkv,Dv,S/2]``, where ``S`` is 32, 64, or
    128. ``block_table[B,max_blocks]`` and ``kv_lengths[B]`` are contiguous
    CUDA int32 tensors. Each physical page/head owns an independent CUTLASS
    SFB scale region.
    """
    if softmax_scale is None:
        logical_head_dim = int(query.shape[-1]) * 2
        softmax_scale = logical_head_dim**-0.5
    return torch.ops.sm120_nvfp4.attention_paged_decode(
        query,
        key_cache,
        value_cache_transposed,
        query_scale,
        key_scale,
        value_scale,
        block_table,
        kv_lengths,
        softmax_scale,
        output,
        workspace,
    )


def grouped_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    output: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run expert-major grouped NVFP4 GEMMs with a single persistent launch."""
    groups = int(weight.shape[0])
    average_rows = int(x.shape[0]) // groups if groups else 0
    return torch.ops.sm120_nvfp4.grouped_gemm(
        x,
        weight,
        seqlens,
        cu_seqlens,
        x_scale,
        weight_scale,
        average_rows,
        output,
        workspace,
    )


def fused_moe(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_weight_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    shared_output: Optional[torch.Tensor] = None,
    ep_rank: int = 0,
    num_experts: Optional[int] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run routing, two grouped GEMMs, activation/quantization and reduction."""
    if num_experts is None:
        num_experts = int(gate_up_weight.shape[0])
    return torch.ops.sm120_nvfp4.fused_moe(
        x,
        x_scale,
        gate_up_weight,
        gate_up_weight_scale,
        down_weight,
        down_weight_scale,
        topk_ids,
        topk_weights,
        shared_output,
        ep_rank,
        num_experts,
        output,
    )


def quantize_expert(
    x: torch.Tensor,
    seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    scale_m_pad: Optional[int] = None,
    output: Optional[torch.Tensor] = None,
    output_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize an expert-major FP16/BF16 matrix to packed NVFP4."""
    if scale_m_pad is None:
        scale_m_pad = max(128, ((int(x.shape[0]) + 127) // 128) * 128)
    return torch.ops.sm120_nvfp4.quantize_expert(
        x, seqlens, cu_seqlens, scale_m_pad, output, output_scale
    )


def expert_moe_workspace_bytes(
    total_rows: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    scale_m_pad: int,
) -> int:
    """Return bytes required by :func:`expert_moe` reusable scratch storage."""
    del hidden_size
    alignment = 256
    total = 0

    def reserve(size: int) -> None:
        nonlocal total
        total = ((total + alignment - 1) // alignment) * alignment
        total += size

    reserve(total_rows * intermediate_size * 4)
    reserve(total_rows * intermediate_size // 2)
    reserve(num_experts * scale_m_pad * scale_k_padded(intermediate_size))
    reserve((num_experts * 3 + 2) * 128)
    reserve((num_experts * 3 + 2) * 128)
    reserve(num_experts * 4)
    reserve((num_experts + 1) * 4)
    return ((total + alignment - 1) // alignment) * alignment


def expert_moe(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_weight_scale: torch.Tensor,
    seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    output: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute already-routed expert-major activations without local re-routing."""
    return torch.ops.sm120_nvfp4.expert_moe(
        x,
        x_scale,
        gate_up_weight,
        gate_up_weight_scale,
        down_weight,
        down_weight_scale,
        seqlens,
        cu_seqlens,
        output,
        workspace,
    )


def scale_k_padded(k: int) -> int:
    return ((k + 15) // 16 + 3) // 4 * 4


def scale_a_elements(m: int, n: int, k: int) -> int:
    del n
    return ((m + 127) // 128 * 128) * scale_k_padded(k)


def scale_b_elements(m: int, n: int, k: int) -> int:
    del m
    return ((n + 127) // 128 * 128) * scale_k_padded(k)


__all__ = [
    "attention_decode",
    "attention_decode_workspace_bytes",
    "attention_paged_decode",
    "attention_prefill",
    "attention_workspace_bytes",
    "expert_moe",
    "expert_moe_workspace_bytes",
    "fused_moe",
    "cute_gemm",
    "cutlass_gemm",
    "gemm",
    "grouped_gemm",
    "quantize_expert",
    "scale_k_padded",
    "scale_a_elements",
    "scale_b_elements",
]
