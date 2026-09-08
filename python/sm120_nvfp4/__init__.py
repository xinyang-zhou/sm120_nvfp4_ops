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


def scale_k_padded(k: int) -> int:
    return ((k + 15) // 16 + 3) // 4 * 4


def scale_a_elements(m: int, n: int, k: int) -> int:
    del n
    return ((m + 127) // 128 * 128) * scale_k_padded(k)


def scale_b_elements(m: int, n: int, k: int) -> int:
    del m
    return ((n + 127) // 128 * 128) * scale_k_padded(k)

__all__ = [
    "fused_moe",
    "cute_gemm",
    "cutlass_gemm",
    "gemm",
    "grouped_gemm",
    "scale_k_padded",
    "scale_a_elements",
    "scale_b_elements",
]
