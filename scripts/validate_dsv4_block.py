#!/usr/bin/env python3
"""GPU-server correctness report for the CSA block. No performance timing."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys

import torch
from sm120_nvfp4 import DSV4CSAAttentionBlock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "python"))
from dsv4_block_reference import full_prefix_reference


def metrics(actual, expected):
    a, b = actual.float(), expected.float()
    delta = a - b
    finite = lambda value: value if math.isfinite(value) else None
    return dict(max_abs=finite(delta.abs().max().item()), rmse=finite(delta.square().mean().sqrt().item()),
                relative_l2=finite((delta.norm() / b.norm().clamp_min(1.e-12)).item()),
                nonfinite=(~torch.isfinite(a) | ~torch.isfinite(b)).sum().item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="local official snapshot; otherwise initialized weights")
    parser.add_argument("--layer-id", type=int, default=2)
    parser.add_argument("--hidden", type=Path, help="optional torch.save BF16 [T,D] input to attention norm")
    parser.add_argument("--tokens", type=int, default=137)
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument("--decode-tail", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1927)
    parser.add_argument("--flashinfer", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if args.tokens <= 0 or args.decode_tail < 0 or args.query_chunk_size <= 0:
        parser.error("invalid token/chunk arguments")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("an SM120 GPU is required; skips are not validation")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.checkpoint:
        block = DSV4CSAAttentionBlock.from_checkpoint(args.checkpoint,
            layer_id=args.layer_id, query_chunk_size=args.query_chunk_size)
    else:
        block = DSV4CSAAttentionBlock(query_chunk_size=args.query_chunk_size)
    if args.hidden:
        hidden = torch.load(args.hidden, map_location="cpu", weights_only=True)
        if not isinstance(hidden, torch.Tensor) or hidden.dtype != torch.bfloat16:
            raise ValueError("hidden file must contain a BF16 tensor")
        hidden = hidden.cuda().contiguous()
    else:
        hidden = torch.randn((args.tokens, block.config.hidden_size), device="cuda", dtype=torch.bfloat16)
    count = hidden.shape[0]
    if count < 1 or args.decode_tail >= count:
        raise ValueError("need at least one prefill token before the decode tail")
    full_state, chunk_state = block.new_state(count), block.new_state(count)
    trace = []
    output = block.prefill(hidden, [full_state], [count], trace=trace)
    expected, details = full_prefix_reference(block, hidden)
    chunk_outputs = []
    prefix_end = count - args.decode_tail
    # Exercise a nonaligned compression boundary independently of the block's
    # internal query tiling. No timing or benchmark is collected.
    position = 0
    for size in (3, 65, 7):
        end = min(prefix_end, position + size)
        if end > position:
            chunk_outputs.append(block.prefill(hidden[position:end], [chunk_state], [end - position]))
        position = end
    if position < prefix_end:
        chunk_outputs.append(block.prefill(hidden[position:prefix_end], [chunk_state], [prefix_end - position]))
    for row in range(prefix_end, count):
        chunk_outputs.append(block.decode(hidden[row:row + 1], [chunk_state]))
    chunk_output = torch.cat(chunk_outputs)
    comparisons = dict(stateless_reference=metrics(output, expected),
                       chunked_then_decode=metrics(chunk_output, output))
    failures = []
    for name, a, b in (("stateless_reference", output, expected), ("chunked_then_decode", chunk_output, output)):
        try:
            torch.testing.assert_close(a, b, rtol=.06, atol=.015)
            if comparisons[name]["rmse"] is None or comparisons[name]["rmse"] > .003:
                raise AssertionError("RMSE exceeds .003")
        except AssertionError as error:
            failures.append(f"{name}: {error}")
    selected = torch.cat([item["compressed_indices"] for item in trace])
    selection_differences = (selected != details["indices"]).sum().item()
    if selection_differences:
        failures.append(f"{selection_differences} selected slot positions differ from stateless reference")
    state_report = {}
    for name, a in full_state.tensors().items():
        b = chunk_state.tensors()[name]
        if a.dtype == torch.uint8:
            different = (a != b).sum().item()
            state_report[name] = dict(different_bytes=different)
            if different:
                failures.append(f"state {name}: {different} bytes differ")
        else:
            state_report[name] = dict(shape=list(a.shape))
            try:
                torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
            except AssertionError as error:
                failures.append(f"state {name}: {error}")
    if args.flashinfer:
        baseline = block.prefill(hidden, [block.new_state(count)], [count], backend="flashinfer")
        comparisons["flashinfer_same_block"] = metrics(output, baseline)
        try:
            torch.testing.assert_close(output, baseline, rtol=.06, atol=.015)
            if comparisons["flashinfer_same_block"]["rmse"] is None or comparisons["flashinfer_same_block"]["rmse"] > .003:
                raise AssertionError("FlashInfer block RMSE exceeds .003")
        except AssertionError as error:
            failures.append(f"flashinfer: {error}")
    report = dict(date_utc=datetime.now(timezone.utc).isoformat(),
        commit=subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
        config=vars(block.config), checkpoint=block.checkpoint_info,
        input_source=str(args.hidden) if args.hidden else "synthetic hidden states",
        seed=args.seed, tokens=count, decode_tail=args.decode_tail, query_chunk_size=args.query_chunk_size,
        precision="native NVFP4 attention/cache; checkpoint projection dtypes; MXFP4 indexer",
        comparisons=comparisons, selected_slot_differences=selection_differences,
        state=state_report, passed=not failures, failures=failures)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(dict(passed=not failures, comparisons=comparisons,
                         selected_slot_differences=selection_differences), indent=2))
    if failures:
        raise SystemExit(f"CSA validation failed; retain {args.output} and stdout/stderr")


if __name__ == "__main__":
    main()
