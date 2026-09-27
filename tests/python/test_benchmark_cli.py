"""CPU-only argument/import regression tests (requires the Python package)."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
import benchmark_attention_ops as attention
from common import operator_benchmark_utils


class BenchmarkCliTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sm120_benchmark_cli_")
        self.addCleanup(self.temp.cleanup)
        self.output = str(Path(self.temp.name) / "result.json")

    def parse(self, *args):
        return attention.parse_args(["--output", self.output, *args])

    def test_native_default_and_explicit_dispatch(self):
        self.assertEqual(self.parse().flashinfer_dispatch, "native")
        self.assertEqual(self.parse("--flashinfer", "--flashinfer-dispatch", "native").kinds,
                         ["csa", "hca"])

    def test_public_csa_decode_and_batch_override(self):
        args = self.parse("--flashinfer", "--flashinfer-dispatch", "public",
                          "--modes", "decode", "--kinds", "csa",
                          "--batches", "64", "512", "1024")
        self.assertEqual(args.batches, [64, 512, 1024])
        self.assertEqual(args.flashinfer_dispatch, "public")

    def test_public_rejects_missing_flag_and_unsupported_workloads(self):
        for extra in ([], ["--modes", "prefill", "--kinds", "csa", "--flashinfer"],
                      ["--modes", "decode", "--kinds", "hca", "--flashinfer"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.parse("--flashinfer-dispatch", "public", *extra)
                self.assertEqual(error.exception.code, 2)

    def test_common_root_survives_directory_move(self):
        self.assertEqual(operator_benchmark_utils.ROOT, ROOT)


if __name__ == "__main__":
    unittest.main()
