#!/usr/bin/env python3
"""DSV4-Flash-shaped CSA decode baseline through FlashInfer's public SM120 API.

Run on the GPU server using a reviewed FlashInfer source revision. No local
operator build is needed. Inputs are synthetic, post-compression/post-RoPE
vectors; neither model weights nor an indexer/compressor are executed here.
Each request has an independent SWA and compressed cache. Only decode is timed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys

import torch
import flashinfer
from flashinfer.mla import (
    nvfp4_quantize_pack_sparse_mla_cache,
    trtllm_batch_decode_sparse_mla_dsv4,
)


# The sparse MLA API, private planner, NVFP4 layout and SM120 kernels are
# unchanged between these revisions. Record the actual revision in every run.
REVIEWED_FLASHINFER_COMMITS = (
    "37b4d30eac39b89f198b893dd11914bd76f5fcf8",
    "ea728cb558c32a3c58ec8fbd5a154ff676b9ab70",
)
HEADS, DIM, SWA, TOPK, COMPRESSION = 64, 512, 128, 512, 4
PAGE_SIZE, PACKED_BYTES = 64, 384


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--context-length", type=int, default=32768,
                        help="History tokens before the current query (multiple of 256).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--eager", action="store_true",
                        help="Time public API calls with CUDA events instead of graph replay.")
    parser.add_argument("--l2-flush-mib", type=int, default=256,
                        help="Zero this buffer before each timed call, outside timing; 0 disables it.")
    parser.add_argument("--output", type=Path,
                        help="Optionally save metadata and raw timing samples as JSON.")
    args = parser.parse_args()
    if any(b <= 0 for b in args.batches) or len(set(args.batches)) != len(args.batches):
        parser.error("batches must be distinct positive integers")
    if args.context_length < TOPK * COMPRESSION or args.context_length % 256:
        parser.error("context-length must be >= 2048 and a multiple of 256")
    if max(args.batches) * (args.context_length // COMPRESSION) >= 2**31:
        parser.error("compressed slot IDs must fit int32")
    if args.warmup < 1 or args.samples < 20 or args.rounds < 1:
        parser.error("require warmup >= 1, samples >= 20, rounds >= 1")
    if args.l2_flush_mib < 0:
        parser.error("l2-flush-mib must be nonnegative")
    if args.output and args.output.exists():
        parser.error("output already exists; choose a new result filename")
    return args


def git_value(directory: Path, *arguments: str) -> str:
    result = subprocess.run(["git", "-C", str(directory), *arguments],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def environment() -> dict:
    source = Path(flashinfer.__file__).resolve().parent.parent
    commit = git_value(source, "rev-parse", "HEAD")
    if commit not in REVIEWED_FLASHINFER_COMMITS:
        raise RuntimeError(
            f"FlashInfer revision {commit} at {source} has not been reviewed "
            f"for this harness. Reviewed revisions: {', '.join(REVIEWED_FLASHINFER_COMMITS)}. "
            "Check sparse MLA API and cache-layout compatibility before adding a revision."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) not in ((12, 0), (12, 1)):
        raise RuntimeError("This baseline requires SM120/SM121")
    patch = git_value(source, "diff", "HEAD", "--", "flashinfer", "include", "csrc")
    if patch:
        raise RuntimeError("FlashInfer runtime sources differ from the checked-out commit")
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "flashinfer": flashinfer.__version__,
        "flashinfer_source": str(source),
        "flashinfer_commit": commit,
        "project_commit": git_value(Path(__file__).resolve().parent, "rev-parse", "HEAD"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "gpu": properties.name,
        "gpu_uuid": str(getattr(properties, "uuid", "unavailable")),
        "capability": [properties.major, properties.minor],
        "gpu_memory_bytes": properties.total_memory,
    }


def synthetic_vectors(shape: tuple, generator: torch.Generator) -> torch.Tensor:
    # Unit RMS reflects normalization at the core boundary, but does not model
    # learned normalization weights or real activation distributions.
    values = torch.randn(shape, generator=generator, device="cuda", dtype=torch.float32)
    values *= torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + 1e-6)
    return values.to(torch.bfloat16)


def build_cache(batch: int, rows: int, seed: int, selected: torch.Tensor,
                check_requests: list[int]) -> tuple[torch.Tensor, dict]:
    pages = rows // PAGE_SIZE
    cache = torch.empty((batch * pages, 1, PAGE_SIZE, PACKED_BYTES),
                        device="cuda", dtype=torch.uint8)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    originals = {}
    # Bound construction memory independently of B/context. The packed cache
    # still reserves every visible compressed row, not just selected rows.
    for first in range(0, batch, 8):
        count = min(8, batch - first)
        source = synthetic_vectors((count * pages, PAGE_SIZE, DIM), generator)
        packed = nvfp4_quantize_pack_sparse_mla_cache(source, kv_layout="HND")
        cache[first * pages:(first + count) * pages].copy_(packed)
        for request in check_requests:
            if first <= request < first + count:
                local = source.view(count, rows, DIM)[request - first]
                originals[request] = local.index_select(0, selected[request].long())
    return cache, originals


def dequantize_selected(cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Read selected rows of the reviewed 384-byte NVFP4 FOOTER ABI."""
    flat = cache.view(cache.shape[0], -1)
    pages, rows = slots.long() // PAGE_SIZE, slots.long() % PAGE_SIZE
    data = flat[pages[:, None], rows[:, None] * 352 + torch.arange(352, device="cuda")]
    scales = flat[pages[:, None], PAGE_SIZE * 352 + rows[:, None] * 32
                  + torch.arange(28, device="cuda")].contiguous()
    codes = torch.stack((data[:, :224] & 15, data[:, :224] >> 4), dim=-1).reshape(-1, 448)
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6],
                       device="cuda", dtype=torch.float32)
    nope = lut[codes.long()] * scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
    rope = data[:, 224:].contiguous().view(torch.bfloat16).float()
    return torch.cat((nope, rope), dim=-1)


def reference(q: torch.Tensor, kv: torch.Tensor, sink: torch.Tensor) -> torch.Tensor:
    logits = q.float() @ kv.float().T / math.sqrt(DIM)
    probabilities = torch.softmax(torch.cat((logits, sink[:, None]), dim=-1), dim=-1)
    return probabilities[:, :-1] @ kv.float()


def errors(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual, expected = actual.float(), expected.float()
    delta = actual - expected
    return {
        "max_abs": delta.abs().max().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "relative_l2": (delta.norm() / expected.norm().clamp_min(1e-12)).item(),
        "cosine_mean": torch.nn.functional.cosine_similarity(
            actual.flatten(0, -2), expected.flatten(0, -2), dim=-1).mean().item(),
    }


def inspect_public_plan(q, swa_cache, swa_ids, output, swa_lengths, sink,
                        comp_cache, comp_ids, comp_lengths):
    # Read the same metadata-keyed plan used by the public facade. This private
    # diagnostic is why this script accepts only reviewed source revisions.
    from flashinfer.mla._sparse_mla_sm120 import _prepared

    tensors = (q[:, 0], swa_cache, swa_ids, output[:, 0], swa_lengths, sink,
               comp_cache, comp_ids, comp_lengths, None, None, None)
    prepared = _prepared._functional_plan(tensors, 1, True, False)
    info = dict(prepared.plan.inspect())
    text_fields = {"numeric_route", "implementation", "merge"}
    return {str(k): str(v) if str(k) in text_fields else int(v) for k, v in info.items()}


def measure(run, args: argparse.Namespace) -> tuple[list[list[float]], torch.Tensor]:
    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    eager_output = run().clone()
    graph = None
    if not args.eager:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        execute = graph.replay
    else:
        execute = run
    flush = (torch.empty(args.l2_flush_mib << 20, dtype=torch.uint8, device="cuda")
             if args.l2_flush_mib else None)
    samples = []
    for _ in range(args.rounds):
        for _ in range(args.warmup):
            execute()
        torch.cuda.synchronize()
        events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                  for _ in range(args.samples)]
        for start, end in events:
            if flush is not None:
                flush.zero_()
            start.record()
            execute()
            end.record()
        torch.cuda.synchronize()
        samples.append([start.elapsed_time(end) * 1000 for start, end in events])
    # The caller checks the output left by the last timed execution against
    # eager_output before doing any further kernel calls.
    return samples, eager_output


def percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * quantile
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def run_case(args: argparse.Namespace, batch: int) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    compressed_rows = args.context_length // COMPRESSION
    check_requests = sorted({0, batch // 2, batch - 1})
    rng = random.Random(args.seed)
    local_comp = torch.tensor([rng.sample(range(compressed_rows), TOPK) for _ in range(batch)],
                              dtype=torch.int32, device="cuda")
    local_swa = torch.arange(SWA, device="cuda", dtype=torch.int32).expand(batch, -1)
    offset = torch.arange(batch, device="cuda", dtype=torch.int32)[:, None]
    swa_ids = (local_swa + offset * SWA).contiguous()
    comp_ids = (local_comp + offset * compressed_rows).contiguous()
    swa_cache, swa_original = build_cache(batch, SWA, args.seed + 1, local_swa, check_requests)
    comp_cache, comp_original = build_cache(batch, compressed_rows, args.seed + 2,
                                          local_comp, check_requests)
    q = synthetic_vectors((batch, 1, HEADS, DIM),
                          torch.Generator(device="cuda").manual_seed(args.seed + 3))
    sink = torch.randn(HEADS, device="cuda", generator=
                       torch.Generator(device="cuda").manual_seed(args.seed + 4))
    swa_lengths = torch.full((batch,), SWA, dtype=torch.int32, device="cuda")
    comp_lengths = torch.full((batch,), TOPK, dtype=torch.int32, device="cuda")
    output = torch.empty_like(q)
    # 64 MiB accommodates B=64, H=64 and 10 split-K tiles at CPB=1;
    # larger batches select streaming with a much smaller scratch requirement.
    workspace = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")

    def run():
        return trtllm_batch_decode_sparse_mla_dsv4(
            query=q, swa_kv_cache=swa_cache, workspace_buffer=workspace,
            sparse_indices=swa_ids, swa_topk_lens=swa_lengths,
            compressed_kv_cache=comp_cache, extra_sparse_indices=comp_ids,
            extra_sparse_topk_lens=comp_lengths, sinks=sink, out=output,
            bmm1_scale=DIM**-0.5, kv_layout="HND", backend="sparse",
            kv_cache_format="nvfp4",
        )

    run()
    torch.cuda.synchronize()
    if not torch.isfinite(output).all().item():
        raise RuntimeError(f"B={batch}: output contains NaN/Inf")
    plan = inspect_public_plan(q, swa_cache, swa_ids, output, swa_lengths, sink,
                               comp_cache, comp_ids, comp_lengths)
    if plan["numeric_route"] != "nvfp4":
        raise RuntimeError(f"Unexpected numeric route: {plan}")

    # Reference diagnostics sample requests across the batch; upstream tests
    # remain the implementation-correctness gate. These errors include Q/P/V
    # arithmetic approximations and are not a model-quality acceptance test.
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        original_refs, cache_refs = [], []
        for request in check_requests:
            original_kv = torch.cat((swa_original[request], comp_original[request]))
            cache_kv = torch.cat((dequantize_selected(swa_cache, swa_ids[request]),
                                  dequantize_selected(comp_cache, comp_ids[request])))
            original_refs.append(reference(q[request, 0], original_kv, sink))
            cache_refs.append(reference(q[request, 0], cache_kv, sink))
        actual = output[check_requests, 0]
        accuracy = {
            "sampled_requests": check_requests,
            "vs_original_bf16": errors(actual, torch.stack(original_refs)),
            "vs_dequantized_kv_bf16_query": errors(actual, torch.stack(cache_refs)),
        }
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    setup_peak = torch.cuda.max_memory_allocated()
    del swa_original, comp_original, original_refs, cache_refs, original_kv, cache_kv, actual
    del local_comp, local_swa, offset
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples, eager_output = measure(run, args)
    if not torch.isfinite(output).all().item():
        raise RuntimeError(f"B={batch}: timed output contains NaN/Inf")
    torch.testing.assert_close(output, eager_output, rtol=0, atol=0)
    final_plan = inspect_public_plan(q, swa_cache, swa_ids, output, swa_lengths, sink,
                                     comp_cache, comp_ids, comp_lengths)
    if final_plan != plan:
        raise RuntimeError("Planner changed during measurement; rerun after calibration settles")
    flattened = [value for run_samples in samples for value in run_samples]
    p50 = statistics.median(flattened)
    memory = {
        "cache_bytes": swa_cache.numel() + comp_cache.numel(),
        "workspace_allocated_bytes": workspace.numel(),
        "measurement_only_flush_bytes": args.l2_flush_mib << 20,
        "torch_peak_setup_bytes": setup_peak,
        "torch_peak_measurement_bytes": torch.cuda.max_memory_allocated(),
    }
    result = {
        "batch": batch, "query_tokens_per_request": 1, "total_query_tokens": batch,
        "history_tokens_per_request": args.context_length,
        "visible_compressed_rows_per_request": compressed_rows,
        "selected_compressed_rows_per_request": TOPK, "swa_rows_per_request": SWA,
        "plan": plan, "accuracy": accuracy, "memory": memory,
        "latency_samples_us": samples,
        "round_p50_us": [statistics.median(values) for values in samples],
        "p50_us": p50, "p95_us": percentile(flattened, .95),
        "core_queries_per_second_at_p50": batch * 1e6 / p50,
    }
    print(f"B={batch:4d}  {plan['implementation']}  cpb={plan['cpb']}  "
          f"p50={p50:.3f} us  p95={result['p95_us']:.3f} us  "
          f"core_q/s={result['core_queries_per_second_at_p50']:.0f}  "
          f"cache={memory['cache_bytes'] / 2**30:.3f} GiB", flush=True)
    original_error = accuracy["vs_original_bf16"]
    cache_error = accuracy["vs_dequantized_kv_bf16_query"]
    print(f"  rounds_us={[round(x, 3) for x in result['round_p50_us']]}  "
          f"checked_requests={check_requests}  "
          f"BF16_ref(max_abs={original_error['max_abs']:.5g}, relL2={original_error['relative_l2']:.5g})  "
          f"dequantKV_ref(max_abs={cache_error['max_abs']:.5g}, relL2={cache_error['relative_l2']:.5g})",
          flush=True)
    return result


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(0)
    info = environment()
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": info,
        "configuration": {
            "model_shape": "DeepSeek-V4-Flash CSA (layer index 2), TP=1",
            "heads": HEADS, "kv_heads": 1, "dim": DIM, "rope_dim": 64,
            "compression": COMPRESSION, "topk": TOPK, "swa": SWA,
            "cache_page_size": PAGE_SIZE, "seed": args.seed,
            "input_kind": "synthetic unit-RMS post-compression/post-RoPE vectors; random sink",
            "indices": "unique random compressed entries; chronological SWA incl. current token",
            "precision": "BF16 Q/output, NVFP4 non-RoPE KV, BF16 RoPE KV",
            "timing": "eager CUDA events" if args.eager else "CUDA Graph replay, one call per sample",
            "cache_policy": "fixed inputs and indices reused; buffer zeroed before each timed call",
            "l2_flush_buffer_bytes": args.l2_flush_mib << 20,
            "boundary": "decode core including in-kernel Q/P/V conversion and merge",
            "excluded": "cache generation/packing, compressor/indexer/Top-K, projection, JIT, calibration",
            "warmup": args.warmup, "samples_per_round": args.samples, "rounds": args.rounds,
        },
        "cases": [],
    }
    print(f"GPU={info['gpu']}  FlashInfer={info['flashinfer_commit'][:12]}  "
          f"history={args.context_length}  B={args.batches}", flush=True)
    print(f"Synthetic CSA core; {report['configuration']['timing']}; "
          f"cache disturbance={args.l2_flush_mib} MiB before each timed call.",
          flush=True)
    for batch in args.batches:
        report["cases"].append(run_case(args, batch))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(f"Saved measurements: {args.output}")


if __name__ == "__main__":
    main()
