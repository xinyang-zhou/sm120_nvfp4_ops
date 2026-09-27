#!/usr/bin/env python3
"""Preallocated NVFP4 cache packing/append: exact bytes, then GPU timing."""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import torch
import sm120_nvfp4

from common.operator_benchmark_utils import (add_timing_arguments, environment,
    flashinfer_environment, measure, run_cases, validate_timing)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "python"))
from sparse_mla_reference import pack_cache


@torch.no_grad()
def case(spec, args):
    rows, operation = spec["rows"], spec["operation"]
    capacity = ((rows * (2 if operation == "append" else 1) + 63) // 64) * 64
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    initial = (torch.randn((capacity, 512), generator=generator, device="cuda") * .25).bfloat16()
    if operation == "pack":
        initial.zero_()
    cache = pack_cache(initial.reshape(-1, 64, 512))
    values = (torch.randn((rows, 512), generator=generator, device="cuda") * .25).bfloat16()
    ids = random.Random(args.seed).sample(range(capacity), rows) if operation == "append" else list(range(rows))
    slots = torch.tensor(ids, device="cuda", dtype=torch.int32)
    expected_values = initial.clone()
    expected_values[slots.long()] = values
    expected = pack_cache(expected_values.reshape(-1, 64, 512))
    fi_cache = cache.clone() if args.flashinfer else None

    def own():
        return (sm120_nvfp4.sparse_mla_pack_cache(values, slots, cache),)

    own()
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)
    result = dict(**spec, shape=[rows, 512], cache_pages=capacity // 64,
        input_bytes=values.numel() * 2 + slots.numel() * 4, cache_bytes=cache.numel(),
        correctness=dict(reference="independent E2M1 codebook + footer layout", different_bytes=0),
        boundary="BF16 to NVFP4 quantization and writes into caller-owned cache",
        selection="unique random physical slots" if operation == "append" else "sequential rows",
        excluded="allocation, input/slot generation, reference, JIT",
        timing=measure(own, args))
    if args.flashinfer:
        from flashinfer.mla import nvfp4_quantize_append_sparse_mla_cache

        def baseline():
            nvfp4_quantize_append_sparse_mla_cache(values, slots, fi_cache)
            return (fi_cache,)

        baseline()
        torch.testing.assert_close(fi_cache, expected, rtol=0, atol=0)
        result["flashinfer"] = dict(api="nvfp4_quantize_append_sparse_mla_cache; preallocated",
                                    different_bytes=0, timing=measure(baseline, args))
        result["speedup_vs_flashinfer"] = result["flashinfer"]["timing"]["p50_us"] / result["timing"]["p50_us"]
    result["rows_per_second"] = rows * 1.e6 / result["timing"]["p50_us"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 16, 128, 1024, 8192])
    parser.add_argument("--operations", nargs="+", choices=("pack", "append"), default=["pack", "append"])
    add_timing_arguments(parser)
    args = parser.parse_args()
    validate_timing(parser, args)
    if min(args.rows) < 1 or max(args.rows) > 1048576:
        parser.error("rows must be in [1,1048576]")
    metadata = environment(__file__)
    if args.flashinfer:
        metadata["flashinfer"] = flashinfer_environment()
    cases = [dict(operation=op, rows=n) for op in args.operations for n in args.rows]
    run_cases(args, metadata, cases, lambda spec: case(spec, args))


if __name__ == "__main__":
    main()
