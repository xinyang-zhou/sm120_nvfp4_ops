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
    """Compute GEMM with M=128/M=256 specialization and generic fallback."""
    return torch.ops.sm120_nvfp4.gemm(a, b, sfa, sfb)


def cute_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
) -> torch.Tensor:
    """Run the generic repository-owned persistent CuTe SM120 kernel."""
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


def sparse_mla_decode_workspace_bytes(
    batch: int,
    swa_candidates: int = 128,
    compressed_candidates: int = 512,
    chunks_per_cta: int = 0,
) -> int:
    """BF16 split outputs + FP32 base-2 split LSE; zero for a single CTA.

    The default uses at most nine chunks/CTA for B<=64, all chunks otherwise.
    This reproduces the initial attention.tex schedule; it is not autotuned.
    """
    if not 1 <= batch <= 1048576:
        raise ValueError("batch must be in [1,1048576]")
    if not (0 <= swa_candidates <= 1048576 and 0 <= compressed_candidates <= 1048576):
        raise ValueError("candidate capacities must be in [0,1048576]")
    if not 0 <= chunks_per_cta <= 2147483647:
        raise ValueError("chunks_per_cta must be a nonnegative int32")
    chunks = max(1, (swa_candidates + 63) // 64 + (compressed_candidates + 63) // 64)
    cpb = min(chunks_per_cta, chunks) if chunks_per_cta else (min(9, chunks) if batch <= 64 else chunks)
    splits = (chunks + cpb - 1) // cpb
    return batch * 64 * splits * (512 * 2 + 4) if splits > 1 else 0


def sparse_mla_decode(
    query: torch.Tensor,
    swa_cache: torch.Tensor,
    compressed_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    compressed_indices: torch.Tensor,
    *,
    swa_lengths: Optional[torch.Tensor] = None,
    compressed_lengths: Optional[torch.Tensor] = None,
    sink: Optional[torch.Tensor] = None,
    chunks_per_cta: int = 0,
    softmax_scale: float = 512**-0.5,
    lse_scale: float = 1.0,
    output: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DS-V4 CSA sparse MLA decode using native C++ CuTe on SM120.

    Q/output: BF16 [B,64,512] or [B,1,64,512], already post-RoPE.
    Both caches: FlashInfer NVFP4 uint8 [P,64,384] (4D HND/NHD also
    accepted). Pages use a data region followed by a scale footer; the
    apparent 384-byte tensor rows are NOT standalone encoded tokens.
    Indices: int32 [B,K] or [B,1,K], flattened physical slots in each pool.
    Optional int32 lengths[B] mask candidate-list suffixes. Negative and
    out-of-range slots are masked; repeated slots count repeatedly.

    Non-RoPE QK/PV use NVFP4, RoPE QK/PV use BF16, softmax/accumulation
    use FP32. The 448-dimensional V is requantized along the gathered
    candidate axis. Sink is optional FP32 [64], finite or -inf, added once
    to the joint denominator. Output LSE is FP32 [B,64], base-2 by default;
    set lse_scale=math.log(2) for natural-log LSE. Empty rows return zero
    output and sink-only LSE (-inf without sink).

    Caller performs selection, compression, RoPE and causal visibility.
    All inputs must be finite except sink=-inf; buffers must not overlap.
    No autograd support. This replaces the former dense/paged decode API.
    """
    return torch.ops.sm120_nvfp4.sparse_mla_decode(
        query, swa_cache, compressed_cache, swa_indices, compressed_indices,
        swa_lengths, compressed_lengths, sink, chunks_per_cta, softmax_scale,
        lse_scale, output, lse, workspace,
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
    "sparse_mla_decode",
    "sparse_mla_decode_workspace_bytes",
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
