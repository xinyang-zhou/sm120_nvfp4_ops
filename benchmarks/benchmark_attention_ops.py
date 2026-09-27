#!/usr/bin/env python3
"""Correctness-gated CSA/HCA decode/prefill core benchmark on an SM120 server."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import sm120_nvfp4

from common.operator_benchmark_utils import (add_timing_arguments, environment, errors,
    flashinfer_environment, measure, run_cases, validate_timing)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "python"))
from attention_workloads import attention_workload
from sparse_mla_reference import plan, reference


@torch.no_grad()
def case(spec, args):
    inputs, kwargs, workload = attention_workload(spec["batch"], spec["queries"],
        args.context_length, spec["kind"], seed=args.seed,
        mixed_lengths=args.mixed_lengths, selection=args.selection)
    q, swa, comp, si, ci = inputs
    rows, capacity = q.shape[0], ci.shape[1]
    prefill = spec["mode"] == "prefill"
    cpb, splits = plan(rows, 128, capacity, 2147483647 if prefill else args.chunks_per_cta)
    output, lse = torch.empty_like(q), torch.empty((rows, 64), device="cuda")
    workspace = torch.empty(0 if prefill else sm120_nvfp4.sparse_mla_decode_workspace_bytes(
        rows, 128, capacity, cpb), device="cuda", dtype=torch.uint8)

    def own():
        if prefill:
            return sm120_nvfp4.sparse_mla_prefill(*inputs, **kwargs, output=output, lse=lse)
        return sm120_nvfp4.sparse_mla_decode(*inputs, **kwargs, chunks_per_cta=cpb,
                                            output=output, lse=lse, workspace=workspace)

    own()
    # Check the first/last request and query boundaries, without materializing
    # the slow reference for the whole timed batch. The unit suite checks all rows.
    check = sorted({0, spec["queries"] - 1, rows // 2, rows - 1})
    ref, ref_lse = reference(q[check], swa, comp, si[check], ci[check],
        swa_lengths=kwargs["swa_lengths"][check],
        compressed_lengths=kwargs["compressed_lengths"][check],
        sink=kwargs["sink"], chunks_per_cta=cpb)
    torch.testing.assert_close(output[check], ref, rtol=.05, atol=.02)
    torch.testing.assert_close(lse[check], ref_lse, rtol=2e-5, atol=5e-4)
    accuracy = errors(output[check], ref)
    if accuracy["rmse"] > .005 or not torch.isfinite(output).all().item():
        raise AssertionError("core reference RMSE exceeds .005 or output is nonfinite")
    result = dict(**spec, workload=workload, reference_query_rows=check,
        backend="sm120_nvfp4.sparse_mla_" + spec["mode"],
        precision="BF16 Q/O, NVFP4 non-RoPE QK/PV, BF16 RoPE, FP32 denominator/accumulator",
        accuracy=accuracy, lse_accuracy=errors(lse[check], ref_lse),
        chunks_per_cta=cpb, active_splits=splits, workspace_bytes=workspace.numel(),
        cache_bytes=swa.numel() + comp.numel(),
        query_output_lse_bytes=q.numel() * 4 + lse.numel() * 4,
        index_length_bytes=(si.numel() + ci.numel() + 2 * rows) * 4,
        boundary="attention core including online Q/P/V conversion and split merge",
        excluded="cache packing, compressor/indexer/selection, RoPE, projections, JIT, reference")

    baseline = None
    public_plan = None
    if args.flashinfer and args.flashinfer_dispatch == "native":
        # Pinned allocation-free native entry: identical candidate order/CPB.
        # This is explicitly NOT the public auto-planner performance baseline.
        from flashinfer.mla._sparse_mla_sm120._dsv4_nvfp4 import _sparse_mla_nvfp4_sm120_paged_attention
        fi_output, fi_lse = torch.empty_like(q), torch.empty_like(lse)
        chunks = (128 + 63) // 64 + (capacity + 63) // 64
        mid_out = None if prefill else torch.empty((rows, 64, chunks, 512), device="cuda", dtype=torch.bfloat16)
        mid_lse = None if prefill else torch.empty((rows, 64, chunks), device="cuda")

        def baseline():
            _sparse_mla_nvfp4_sm120_paged_attention(q, swa, si, fi_output, fi_lse, 512**-.5,
                topk_length=kwargs["swa_lengths"], attn_sink=kwargs["sink"],
                extra_kv_cache=comp, extra_indices=ci, extra_topk_length=kwargs["compressed_lengths"],
                mid_out=mid_out, mid_lse=mid_lse, use_prefill=prefill, chunks_per_block_override=cpb)
            return fi_output, fi_lse

        baseline()
        torch.testing.assert_close(output, fi_output, rtol=.07, atol=.03)
        torch.testing.assert_close(lse, fi_lse, rtol=2e-5, atol=5e-4)
        difference = errors(output, fi_output)
        if difference["rmse"] > .005:
            raise AssertionError("FlashInfer matched-schedule RMSE exceeds .005")
        result["flashinfer"] = dict(accuracy_vs_own=difference,
            dispatch="native", lse_compared=True,
            schedule="native prefill" if prefill else "native decode, explicit matched CPB",
            workspace_bytes=0 if prefill else mid_out.numel() * 2 + mid_lse.numel() * 4)
    elif args.flashinfer:
        # CSA single-token decode only. The public dispatcher owns its CPB;
        # do not label this comparison as a matched-schedule/native baseline.
        from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4
        from common.flashinfer_public import inspect_public_plan

        fi_output = torch.empty_like(q)
        fi_workspace = torch.empty(64 << 20, device="cuda", dtype=torch.uint8)
        public_q, public_out = q[:, None], fi_output[:, None]
        public_swa, public_comp = swa[:, None], comp[:, None]

        def baseline():
            trtllm_batch_decode_sparse_mla_dsv4(
                query=public_q, swa_kv_cache=public_swa, workspace_buffer=fi_workspace,
                sparse_indices=si, swa_topk_lens=kwargs["swa_lengths"],
                compressed_kv_cache=public_comp, extra_sparse_indices=ci,
                extra_sparse_topk_lens=kwargs["compressed_lengths"], sinks=kwargs["sink"],
                out=public_out, bmm1_scale=512**-.5, kv_layout="HND", backend="sparse",
                kv_cache_format="nvfp4")
            return (fi_output,)

        def public_plan():
            return inspect_public_plan(public_q, public_swa, si, public_out,
                kwargs["swa_lengths"], kwargs["sink"], public_comp, ci,
                kwargs["compressed_lengths"])

        baseline()
        torch.testing.assert_close(output, fi_output, rtol=.07, atol=.03)
        difference = errors(output, fi_output)
        if difference["rmse"] > .005:
            raise AssertionError("FlashInfer public-dispatch RMSE exceeds .005")
        selected = public_plan()
        if selected["numeric_route"] != "nvfp4":
            raise RuntimeError(f"Unexpected public numeric route: {selected}")
        result["flashinfer"] = dict(accuracy_vs_own=difference,
            dispatch="public", schedule="public sparse NVFP4 auto planner",
            plan=selected, workspace_bytes=fi_workspace.numel(), lse_compared=False,
            lse_note="The public API does not return LSE; own LSE is checked against the reference")

    # Alternate which backend is measured first across cases; no profiler runs
    # belong in this timing sweep. Keep the same L2 disturbance and graph mode.
    order = [("own", own)] + ([("flashinfer", baseline)] if baseline else [])
    if spec["measure_baseline_first"]:
        order.reverse()
    for name, run in order:
        timing = measure(run, args)
        if name == "own":
            result["timing"] = timing
        else:
            result[name]["timing"] = timing
    result["core_queries_per_second"] = rows * 1.e6 / result["timing"]["p50_us"]
    if baseline:
        result["speedup_vs_flashinfer"] = result["flashinfer"]["timing"]["p50_us"] / result["timing"]["p50_us"]
    if public_plan is not None and public_plan() != result["flashinfer"]["plan"]:
        raise RuntimeError("Public planner changed during timing; rerun after calibration settles")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=("decode", "prefill"), default=["decode", "prefill"])
    parser.add_argument("--kinds", nargs="+", choices=("csa", "hca"), default=["csa", "hca"])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 16, 64])
    parser.add_argument("--prefill-batches", nargs="+", type=int, default=[1])
    parser.add_argument("--query-lengths", nargs="+", type=int, default=[7, 128])
    parser.add_argument("--context-length", type=int, default=8192,
                        help="history before the first query, in original tokens")
    parser.add_argument("--mixed-lengths", action="store_true")
    parser.add_argument("--selection", choices=("random", "contiguous"), default="random")
    parser.add_argument("--chunks-per-cta", type=int, default=0)
    parser.add_argument("--flashinfer-dispatch", choices=("native", "public"), default="native",
        help="with --flashinfer: matched-CPB native kernels (default), or public auto planner (CSA decode only)")
    add_timing_arguments(parser)
    args = parser.parse_args(argv)
    validate_timing(parser, args)
    if (min(args.batches + args.prefill_batches + args.query_lengths) < 1 or
            args.context_length < 0 or args.context_length + max(args.query_lengths) > 1048576 or
            max(args.batches + [b * q for b in args.prefill_batches for q in args.query_lengths]) > 1048576 or
            not 0 <= args.chunks_per_cta <= 2147483647):
        parser.error("invalid request/query/context/cpb sizes")
    if args.flashinfer_dispatch == "public":
        if not args.flashinfer:
            parser.error("--flashinfer-dispatch public requires --flashinfer")
        if args.modes != ["decode"] or args.kinds != ["csa"]:
            parser.error("public dispatch requires --modes decode --kinds csa")
    return args


def main():
    args = parse_args()
    metadata = environment(__file__)
    if args.flashinfer:
        metadata["flashinfer"] = flashinfer_environment()
    cases = []
    for kind in args.kinds:
        for mode in args.modes:
            for batch in args.batches if mode == "decode" else args.prefill_batches:
                for queries in [1] if mode == "decode" else args.query_lengths:
                    cases.append(dict(kind=kind, mode=mode, batch=batch, queries=queries,
                                      measure_baseline_first=bool(len(cases) % 2)))
    run_cases(args, metadata, cases, lambda spec: case(spec, args))


if __name__ == "__main__":
    main()
