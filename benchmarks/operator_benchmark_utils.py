"""GPU-server benchmark support; no model runtime or implicit CPU fallback."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
REVIEWED_FLASHINFER = {
    "37b4d30eac39b89f198b893dd11914bd76f5fcf8",
    "ea728cb558c32a3c58ec8fbd5a154ff676b9ab70",
}


def git_value(path, *args):
    result = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def environment(script):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("Run on the SM120 server; CPU execution/skips are not validation")
    torch.backends.cuda.matmul.allow_tf32 = False
    props = torch.cuda.get_device_properties(0)
    sources = [ROOT / "CMakeLists.txt", ROOT / "pyproject.toml"]
    for directory in ("include", "src", "bindings", "python", "tests", "benchmarks", "scripts"):
        sources.extend(p for p in (ROOT / directory).rglob("*")
                       if p.is_file() and p.suffix in (".py", ".cu", ".cuh", ".hpp", ".cpp", ".sh"))
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        project_commit=git_value(ROOT, "rev-parse", "HEAD"),
        working_tree=git_value(ROOT, "status", "--porcelain"),
        script_sha256=hashlib.sha256(Path(script).read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(sources)},
        command=[sys.executable, *sys.argv], python=sys.version,
        torch=torch.__version__, cuda=torch.version.cuda, gpu=props.name,
        gpu_uuid=str(getattr(props, "uuid", "unknown")), gpu_memory_bytes=props.total_memory,
        capability=[props.major, props.minor])


def flashinfer_environment():
    import flashinfer
    source = Path(flashinfer.__file__).resolve().parent.parent
    commit = git_value(source, "rev-parse", "HEAD")
    if commit not in REVIEWED_FLASHINFER:
        raise RuntimeError(f"Unreviewed FlashInfer revision {commit}; review private APIs before A/B")
    dirty = git_value(source, "diff", "HEAD", "--", "flashinfer", "csrc", "include")
    if dirty:
        raise RuntimeError("FlashInfer runtime sources are modified; use a clean reviewed checkout")
    return dict(version=flashinfer.__version__, commit=commit, source=str(source))


def add_timing_arguments(parser):
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--l2-flush-mib", type=int, default=256)
    parser.add_argument("--flashinfer", action="store_true")
    parser.add_argument("--output", type=Path, required=True)


def validate_timing(parser, args):
    if args.output.exists():
        parser.error("output already exists; select a new result filename")
    if args.warmup < 1 or args.rounds < 3 or args.samples < 20 or args.l2_flush_mib < 0:
        parser.error("require warmup>=1, rounds>=3, samples>=20 and l2-flush-mib>=0")


def percentile(values, fraction):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * fraction
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def measure(run, args):
    """run returns caller-owned output tensors; check graph/eager equivalence."""
    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    live = tuple(run())
    expected = tuple(x.clone() for x in live)
    if args.eager:
        execute = run
    else:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        execute = graph.replay
    for _ in range(args.warmup):
        execute()
    flush = (torch.empty(args.l2_flush_mib << 20, device="cuda", dtype=torch.uint8)
             if args.l2_flush_mib else None)
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.rounds):
        events = []
        for _ in range(args.samples):
            if flush is not None:
                flush.zero_()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            execute()
            end.record()
            events.append((start, end))
        torch.cuda.synchronize()
        samples.append([a.elapsed_time(b) * 1000 for a, b in events])
    for actual, baseline in zip(live, expected):
        torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
    flat = [value for group in samples for value in group]
    if not all(math.isfinite(v) and v > 0 for v in flat):
        raise RuntimeError("Invalid CUDA event durations")
    return dict(p50_us=percentile(flat, .5), p95_us=percentile(flat, .95),
                round_p50_us=[statistics.median(x) for x in samples], samples_us=samples,
                graph_replay_matches_eager=True)


def errors(actual, expected):
    a, b = actual.float(), expected.float()
    finite = torch.isfinite(a) & torch.isfinite(b)
    difference = a[finite] - b[finite]
    return dict(max_abs=difference.abs().max().item() if difference.numel() else None,
        rmse=difference.square().mean().sqrt().item() if difference.numel() else None,
        relative_l2=(difference.norm() / b[finite].norm().clamp_min(1.e-30)).item(),
        actual_nonfinite=(~torch.isfinite(a)).sum().item(),
        reference_nonfinite=(~torch.isfinite(b)).sum().item(),
        nonfinite_mismatches=((~finite) & ~(a == b)).sum().item())


def run_cases(args, metadata, cases, run_case):
    """Persist each finished case and the failing case instead of losing a sweep."""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(environment=metadata,
        configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        timing="CUDA events around eager call" if args.eager else "CUDA events around one CUDA Graph replay",
        cache_policy="fixed inputs; L2 disturbance before each call, outside event interval",
        status="running", cases=[])
    with args.output.open("x") as handle:
        def save():
            content = json.dumps(report, indent=2, allow_nan=False)
            handle.seek(0)
            handle.write(content + "\n")
            handle.truncate()
            handle.flush()
        save()
        try:
            for case in cases:
                report["active_case"] = case
                save()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                result = run_case(case)
                result["peak_allocated_bytes_including_setup_reference"] = torch.cuda.max_memory_allocated()
                report["cases"].append(result)
                save()
                print(json.dumps({"case": case, "status": "passed"}), flush=True)
            report.pop("active_case", None)
            report["status"] = "passed"
        except BaseException:
            report["status"] = "failed"
            report["failure"] = traceback.format_exc()
            raise
        finally:
            save()
