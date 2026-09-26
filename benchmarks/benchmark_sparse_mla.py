#!/usr/bin/env python3
"""SM120 DS-V4 CSA core: self-contained synthetic workload, optional FlashInfer A/B.

Records raw samples; no performance claim is made until this runs on the server.
Cache construction, reference evaluation, JIT and graph capture are untimed.
Both implementations receive the same packed caches, indices, BF16 Q and sink.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys

import torch
import sm120_nvfp4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "python"))
from sparse_mla_reference import pack_cache, plan, reference


def git_value(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], capture_output=True,
                          text=True).stdout.strip() or "unknown"


def percentile(values, fraction):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * fraction
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def unit_vectors(shape, generator):
    x = torch.randn(shape, generator=generator, device="cuda")
    return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1.e-6)).to(torch.bfloat16)


def build_cache(batch, rows, generator):
    cache = torch.empty((batch * rows // 64, 64, 384), device="cuda", dtype=torch.uint8)
    for first in range(0, batch, 4):
        count = min(4, batch - first)
        values = unit_vectors((count * rows // 64, 64, 512), generator)
        cache[first * rows // 64:(first + count) * rows // 64].copy_(pack_cache(values))
    return cache


def time_call(run, args, flush):
    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    live_output = run()
    eager = live_output.clone()
    if args.eager:
        execute = run
    else:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        execute = graph.replay
    for _ in range(args.warmup):
        execute()
    torch.cuda.synchronize()
    rounds = []
    for _ in range(args.rounds):
        events = []
        for _ in range(args.samples):
            if flush is not None:
                flush.zero_()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            execute()
            end.record()
            events.append((start, end))
        torch.cuda.synchronize()
        rounds.append([a.elapsed_time(b) * 1000 for a, b in events])
    torch.testing.assert_close(live_output, eager, rtol=0, atol=0)
    return rounds


def case(batch, args, flashinfer_run):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    rows = args.context_length // 4
    swa_cache = build_cache(batch, 128, generator)
    comp_cache = build_cache(batch, rows, generator)
    offsets = torch.arange(batch, device="cuda", dtype=torch.int32)[:, None]
    swa_ids = (torch.arange(128, device="cuda", dtype=torch.int32)[None] + offsets * 128).contiguous()
    rng = random.Random(args.seed)
    comp_ids = torch.tensor([rng.sample(range(rows), 512) for _ in range(batch)],
                            dtype=torch.int32, device="cuda") + offsets * rows
    q = unit_vectors((batch, 1, 64, 512), generator)
    sink = torch.randn(64, generator=generator, device="cuda")
    output = torch.empty_like(q)
    lse = torch.empty((batch, 64), dtype=torch.float32, device="cuda")
    cpb, splits = plan(batch, 128, 512, args.chunks_per_cta)
    scratch = torch.empty(sm120_nvfp4.sparse_mla_decode_workspace_bytes(batch, 128, 512, cpb),
                          dtype=torch.uint8, device="cuda")
    def own():
        return sm120_nvfp4.sparse_mla_decode(q, swa_cache, comp_cache, swa_ids, comp_ids,
            sink=sink, chunks_per_cta=cpb, output=output, lse=lse, workspace=scratch)[0]
    own()
    check = sorted({0, batch // 2, batch - 1})
    expected, expected_lse = reference(q[check], swa_cache, comp_cache, swa_ids[check], comp_ids[check],
                                      sink=sink, chunks_per_cta=cpb)
    error = output[check].float() - expected.float()
    torch.testing.assert_close(output[check], expected, rtol=.05, atol=.02)
    torch.testing.assert_close(lse[check], expected_lse, rtol=2e-5, atol=5e-4)
    rmse = error.square().mean().sqrt().item()
    if rmse > .005:
        raise RuntimeError(f"native reference RMSE {rmse} exceeds .005")
    flush = torch.empty(args.l2_flush_mib << 20, device="cuda", dtype=torch.uint8) if args.l2_flush_mib else None
    own_samples = time_call(own, args, flush)
    flat = [value for round_ in own_samples for value in round_]
    result = dict(batch=batch, heads=64, dim=512, swa=128, compressed_topk=512,
                  compressed_rows=rows, chunks_per_cta=cpb, splits=splits,
                  reference_requests=check, max_abs_error=error.abs().max().item(), rmse=rmse,
                  p50_us=percentile(flat, .5), p95_us=percentile(flat, .95), samples_us=own_samples,
                  workspace_bytes=scratch.numel(), cache_bytes=swa_cache.numel() + comp_cache.numel())
    if flashinfer_run is not None:
        fi_out = torch.empty_like(q)
        fi_scratch = torch.empty(64 << 20, device="cuda", dtype=torch.uint8)
        swa_len = torch.full((batch,), 128, device="cuda", dtype=torch.int32)
        comp_len = torch.full((batch,), 512, device="cuda", dtype=torch.int32)
        def baseline():
            return flashinfer_run(query=q, swa_kv_cache=swa_cache[:, None], workspace_buffer=fi_scratch,
                sparse_indices=swa_ids, swa_topk_lens=swa_len,
                compressed_kv_cache=comp_cache[:, None], extra_sparse_indices=comp_ids,
                extra_sparse_topk_lens=comp_len, sinks=sink, out=fi_out,
                bmm1_scale=512**-.5, kv_layout="HND", backend="sparse", kv_cache_format="nvfp4")
        baseline()
        # Public FlashInfer chooses its own CTA/split schedule. Its output may
        # differ at quantizer boundaries; the controlled-CPB test is separate.
        torch.testing.assert_close(output, fi_out, rtol=.07, atol=.03)
        samples = time_call(baseline, args, flush)
        flat_fi = [value for round_ in samples for value in round_]
        result["flashinfer"] = dict(p50_us=percentile(flat_fi, .5), p95_us=percentile(flat_fi, .95),
                                   samples_us=samples, workspace_bytes=fi_scratch.numel(),
                                   max_abs_difference=(output.float() - fi_out.float()).abs().max().item(),
                                   schedule="public sparse NVFP4 dispatcher; use profiler to record kernel/CPB")
    result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", type=int, default=[64, 256, 1024])
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--chunks-per-cta", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--l2-flush-mib", type=int, default=256)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--flashinfer", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if (min(args.batches) <= 0 or args.context_length < 2048 or args.context_length % 256 or
            max(args.batches) * (args.context_length // 4) >= 2**31 or args.chunks_per_cta < 0 or
            args.rounds < 1 or args.samples < 20 or args.warmup < 1 or args.l2_flush_mib < 0):
        parser.error("invalid shape, timing, or cache arguments")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("SM120 GPU server required")
    torch.backends.cuda.matmul.allow_tf32 = False
    metadata = dict(date_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
                    torch=torch.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(), project_commit=git_value(ROOT, "rev-parse", "HEAD"),
                    working_tree=git_value(ROOT, "status", "--porcelain"),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    timing="CUDA events, eager" if args.eager else "CUDA events, graph replay",
                    l2_flush_mib=args.l2_flush_mib, warmup=args.warmup, seed=args.seed,
                    lse_base=2, scope="synthetic post-compression/post-RoPE attention core")
    flashinfer_run = None
    if args.flashinfer:
        import flashinfer
        from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4
        source = Path(flashinfer.__file__).resolve().parent.parent
        commit = git_value(source, "rev-parse", "HEAD")
        reviewed = {"37b4d30eac39b89f198b893dd11914bd76f5fcf8", "ea728cb558c32a3c58ec8fbd5a154ff676b9ab70"}
        if commit not in reviewed:
            raise RuntimeError(f"Unreviewed FlashInfer revision: {commit}; verify cache/API before A/B")
        metadata.update(flashinfer_version=flashinfer.__version__, flashinfer_commit=commit,
                        flashinfer_source=str(source))
        flashinfer_run = trtllm_batch_decode_sparse_mla_dsv4
    results = []
    for batch in args.batches:
        entry = case(batch, args, flashinfer_run)
        results.append(entry)
        print(f"B={batch}: P50={entry['p50_us']:.3f} us, P95={entry['p95_us']:.3f} us", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(dict(environment=metadata, results=results), handle, indent=2)


if __name__ == "__main__":
    main()
