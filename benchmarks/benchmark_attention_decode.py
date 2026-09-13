#!/usr/bin/env python3
"""Benchmark dense and paged SM120 NVFP4 single-token decode attention."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch

import sm120_nvfp4


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def time_us(
    function: Callable[[], None], warmup: int, iterations: int
) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def make_dense_cache(
    paged: torch.Tensor, batch: int, max_blocks: int
) -> torch.Tensor:
    _, kv_heads, block_size, packed_head_dim = paged.shape
    return (
        paged.reshape(
            batch, max_blocks, kv_heads, block_size, packed_head_dim
        )
        .permute(0, 2, 1, 3, 4)
        .contiguous()
        .reshape(batch, kv_heads, max_blocks * block_size, packed_head_dim)
    )


def make_dense_value_cache(
    paged: torch.Tensor, batch: int, max_blocks: int
) -> torch.Tensor:
    _, kv_heads, value_dim, packed_block_size = paged.shape
    return (
        paged.reshape(
            batch,
            max_blocks,
            kv_heads,
            value_dim,
            packed_block_size,
        )
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .reshape(
            batch,
            kv_heads,
            value_dim,
            max_blocks * packed_block_size,
        )
    )


def run_case(args: argparse.Namespace, batch: int, block_size: int) -> dict:
    if args.max_kv_length % block_size != 0:
        raise ValueError("max-kv-length must be divisible by every block size")

    torch.manual_seed(args.seed + batch * 1000 + block_size)
    device = torch.device("cuda", args.device)
    max_blocks = args.max_kv_length // block_size
    physical_blocks = batch * max_blocks

    query = torch.randint(
        0,
        256,
        (batch, args.query_heads, args.head_dim // 2),
        dtype=torch.uint8,
        device=device,
    )
    paged_key = torch.randint(
        0,
        256,
        (
            physical_blocks,
            args.kv_heads,
            block_size,
            args.head_dim // 2,
        ),
        dtype=torch.uint8,
        device=device,
    )
    paged_value = torch.randint(
        0,
        256,
        (
            physical_blocks,
            args.kv_heads,
            args.value_dim,
            block_size // 2,
        ),
        dtype=torch.uint8,
        device=device,
    )
    dense_key = make_dense_cache(paged_key, batch, max_blocks)
    dense_value = make_dense_value_cache(paged_value, batch, max_blocks)
    block_table = torch.arange(
        physical_blocks, dtype=torch.int32, device=device
    ).reshape(batch, max_blocks)
    lengths = torch.full(
        (batch,), args.max_kv_length, dtype=torch.int32, device=device
    )

    query_scale = torch.full(
        (
            batch
            * args.query_heads
            * sm120_nvfp4.scale_a_elements(
                1, args.max_kv_length, args.head_dim
            ),
        ),
        args.scale,
        dtype=torch.uint8,
        device=device,
    )
    dense_key_scale = torch.full(
        (
            batch
            * args.kv_heads
            * sm120_nvfp4.scale_b_elements(
                1, args.max_kv_length, args.head_dim
            ),
        ),
        args.scale,
        dtype=torch.uint8,
        device=device,
    )
    dense_value_scale = torch.full(
        (
            batch
            * args.kv_heads
            * sm120_nvfp4.scale_b_elements(
                1, args.value_dim, args.max_kv_length
            ),
        ),
        args.scale,
        dtype=torch.uint8,
        device=device,
    )
    paged_key_scale = torch.full(
        (
            physical_blocks
            * args.kv_heads
            * sm120_nvfp4.scale_b_elements(
                1, block_size, args.head_dim
            ),
        ),
        args.scale,
        dtype=torch.uint8,
        device=device,
    )
    paged_value_scale = torch.full(
        (
            physical_blocks
            * args.kv_heads
            * sm120_nvfp4.scale_b_elements(
                1, args.value_dim, block_size
            ),
        ),
        args.scale,
        dtype=torch.uint8,
        device=device,
    )
    workspace = torch.empty(
        sm120_nvfp4.attention_decode_workspace_bytes(
            batch,
            args.query_heads,
            args.kv_heads,
            args.max_kv_length,
            args.head_dim,
            args.value_dim,
        ),
        dtype=torch.uint8,
        device=device,
    )
    dense_output = torch.empty(
        (batch, args.query_heads, args.value_dim),
        dtype=torch.float16,
        device=device,
    )
    paged_output = torch.empty_like(dense_output)

    def dense() -> None:
        sm120_nvfp4.attention_decode(
            query,
            dense_key,
            dense_value,
            query_scale,
            dense_key_scale,
            dense_value_scale,
            kv_lengths=lengths,
            output=dense_output,
            workspace=workspace,
        )

    def paged() -> None:
        sm120_nvfp4.attention_paged_decode(
            query,
            paged_key,
            paged_value,
            query_scale,
            paged_key_scale,
            paged_value_scale,
            block_table,
            lengths,
            output=paged_output,
            workspace=workspace,
        )

    dense()
    paged()
    torch.cuda.synchronize()
    torch.testing.assert_close(paged_output, dense_output, rtol=0, atol=0)

    dense_us = time_us(dense, args.warmup, args.iterations)
    paged_us = time_us(paged, args.warmup, args.iterations)
    result = {
        "batch": batch,
        "block_size": block_size,
        "dense_us": dense_us,
        "paged_us": paged_us,
        "paged_over_dense": paged_us / dense_us,
        "max_abs_diff": 0.0,
    }
    print(
        f"B={batch} S={block_size}: dense={dense_us:.2f} us, "
        f"paged={paged_us:.2f} us, ratio={paged_us / dense_us:.2f}x"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batches", type=parse_int_list, default=[1, 8])
    parser.add_argument(
        "--block-sizes", type=parse_int_list, default=[32, 64, 128]
    )
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--max-kv-length", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--scale", type=lambda value: int(value, 0), default=0x38)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.query_heads % args.kv_heads != 0:
        raise ValueError("query-heads must be divisible by kv-heads")

    torch.cuda.set_device(args.device)
    results = [
        run_case(args, batch, block_size)
        for batch in args.batches
        for block_size in args.block_sizes
    ]
    report = {
        "date_utc": datetime.now(timezone.utc).date().isoformat(),
        "hardware": torch.cuda.get_device_name(args.device),
        "device": args.device,
        "pytorch_cuda": torch.version.cuda,
        "measurement": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "timer": "CUDA events",
            "validation": "paged output equals dense output elementwise",
        },
        "shape": {
            "query_heads": args.query_heads,
            "kv_heads": args.kv_heads,
            "max_kv_length": args.max_kv_length,
            "head_dim": args.head_dim,
            "value_dim": args.value_dim,
        },
        "results": results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
