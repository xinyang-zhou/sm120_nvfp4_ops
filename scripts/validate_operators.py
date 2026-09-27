#!/usr/bin/env python3
"""Run operator correctness gates on the GPU server and retain pass/fail/skip."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "tests" / "python"))
from operator_benchmark_utils import environment, flashinfer_environment

MODULES = ("test_gemm", "test_grouped_gemm", "test_fused_moe", "test_sparse_mla",
           "test_sparse_mla_prefill", "test_sparse_mla_cache", "test_sparse_mla_workloads")


class Results(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed.append(test.id())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashinfer", action="store_true", help="require pinned FlashInfer integration tests")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status="running", suite="operators", modules=list(MODULES))
    with args.output.open("x") as handle:
        try:
            report["environment"] = environment(__file__)
            # Do not inherit optional-test flags invisibly from the shell.
            os.environ["SPARSE_MLA_FLASHINFER_TEST"] = "1" if args.flashinfer else "0"
            if args.flashinfer:
                report["flashinfer"] = flashinfer_environment()
            suite = unittest.defaultTestLoader.loadTestsFromNames(MODULES)
            result = unittest.TextTestRunner(verbosity=2, resultclass=Results).run(suite)
            unexpected_skips = [(test.id(), reason) for test, reason in result.skipped
                if args.flashinfer or ".test_flashinfer_" not in test.id()]
            passed = result.wasSuccessful() and not unexpected_skips and bool(result.passed)
            report.update(status="passed" if passed else "failed", tests_run=result.testsRun,
                passed=result.passed, failures=[(t.id(), msg) for t, msg in result.failures],
                errors=[(t.id(), msg) for t, msg in result.errors],
                skipped=[(t.id(), why) for t, why in result.skipped],
                unexpected_skips=unexpected_skips)
            if not passed:
                raise RuntimeError("Operator gate failed; preserve JSON and complete stdout/stderr")
        except BaseException:
            report["status"] = "failed"
            report["failure"] = traceback.format_exc()
            raise
        finally:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")


if __name__ == "__main__":
    main()
