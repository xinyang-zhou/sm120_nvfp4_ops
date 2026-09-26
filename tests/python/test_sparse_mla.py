"""Run on an SM120 GPU server; local development does not execute these tests."""
import math
import os
import unittest

import torch
import sm120_nvfp4
from sparse_mla_reference import make_problem, pack_cache, reference


class SparseMlaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
            raise unittest.SkipTest("SM120 required")
        cls.old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    @classmethod
    def tearDownClass(cls):
        torch.backends.cuda.matmul.allow_tf32 = cls.old_tf32

    def compare(self, problem, **kwargs):
        expected, expected_lse = reference(*problem, **kwargs)
        actual, lse = sm120_nvfp4.sparse_mla_decode(*problem, **kwargs)
        torch.cuda.synchronize()
        self.assertEqual(actual.dtype, torch.bfloat16)
        # Allow FP32 reduction/MMA order and quantizer boundary differences;
        # gate both elementwise output and aggregate error, not just cosine.
        torch.testing.assert_close(actual, expected, rtol=.04, atol=.006)
        self.assertLess((actual.float() - expected.float()).square().mean().sqrt().item(), .0015)
        torch.testing.assert_close(lse, expected_lse, rtol=2e-5, atol=3e-4)
        return actual, lse

    def test_csa_128_plus_512_and_split_rounding(self):
        problem = make_problem()
        sink = torch.linspace(-2., 5., 64, device="cuda")
        for cpb in (1, 3, 9, 10):
            with self.subTest(cpb=cpb):
                self.compare(problem, sink=sink, chunks_per_cta=cpb)

    def test_tail_masks_repeated_ids_and_lengths(self):
        problem = list(make_problem(swa=71, compressed=137))
        problem[3][:, 2] = -1
        problem[3][:, 3] = problem[3][:, 1]
        problem[4][:, 7] = problem[2].shape[0] * 64  # out of range
        problem[4][:, 8] = -9
        for cpb in (1, 2, 10):
            self.compare(problem, swa_lengths=torch.tensor([33, 1000], device="cuda", dtype=torch.int32),
                         compressed_lengths=torch.tensor([129, -1], device="cuda", dtype=torch.int32),
                         chunks_per_cta=cpb, lse_scale=math.log(2))

    def test_empty_candidates_empty_splits_and_sink(self):
        problem = make_problem(batch=1, swa=64, compressed=128)
        problem[3].fill_(-1)
        problem[4].fill_(-1)
        for cpb in (1, 3):
            for sink in (None, torch.linspace(-3., 3., 64, device="cuda"),
                         torch.full((64,), -torch.inf, device="cuda")):
                out, lse = self.compare(problem, sink=sink, chunks_per_cta=cpb)
                torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)
                expected = torch.full_like(lse, -torch.inf) if sink is None else sink[None] * math.log2(math.e)
                torch.testing.assert_close(lse, expected, rtol=2e-6, atol=2e-6)
        # One valid candidate in the last split, all previous splits empty.
        problem[4][0, -1] = 0
        self.compare(problem, chunks_per_cta=1)

    def test_zero_capacity_pools_and_zero_scale(self):
        problem = list(make_problem(batch=1, swa=0, compressed=0))
        problem[1] = torch.empty((0, 64, 384), dtype=torch.uint8, device="cuda")
        problem[2] = torch.empty_like(problem[1])
        self.compare(problem)
        problem = list(make_problem(batch=1, swa=64, compressed=0))
        problem[0].zero_()
        latent = torch.full((1, 64, 512), 1.e-7, dtype=torch.bfloat16, device="cuda")
        latent[..., 448:] = .125
        problem[1] = pack_cache(latent)
        problem[3].zero_()
        self.compare(problem, sink=torch.zeros(64, device="cuda"))

    def test_fp32_denominator_and_rope_precision(self):
        problem = list(make_problem(batch=1, swa=64, compressed=0))
        problem[0].zero_()
        problem[1] = pack_cache(torch.ones((1, 64, 512), dtype=torch.bfloat16, device="cuda"))
        problem[3][0] = torch.arange(64, device="cuda", dtype=torch.int32)
        out, lse = self.compare(problem)
        # W=1 becomes 1.03125 after NVFP4 quantization. The denominator is
        # still 64, NOT 64*1.03125. RoPE PV consumes BF16 W=1 directly.
        torch.testing.assert_close(out[..., :448], torch.full_like(out[..., :448], 1.0625), rtol=0, atol=0)
        torch.testing.assert_close(out[..., 448:], torch.ones_like(out[..., 448:]), rtol=0, atol=0)
        torch.testing.assert_close(lse, torch.full_like(lse, 6.), rtol=0, atol=0)

    def test_large_scores_and_sink_dominance(self):
        problem = list(make_problem(batch=1, seed=81))
        problem[0].mul_(16)
        self.compare(problem, chunks_per_cta=3,
                     sink=torch.linspace(-80., 80., 64, device="cuda"))

    def test_reusable_buffers_and_cuda_graph(self):
        q, c1, c2, i1, i2 = make_problem(batch=2, swa=71, compressed=137)
        problem = (q[:, None], c1[:, None], c2[:, :, None], i1[:, None], i2[:, None])
        output = torch.empty_like(problem[0])
        lse = torch.empty((2, 64), dtype=torch.float32, device="cuda")
        scratch = torch.empty(sm120_nvfp4.sparse_mla_decode_workspace_bytes(2, 71, 137, 2),
                              dtype=torch.uint8, device="cuda")
        kwargs = dict(chunks_per_cta=2, output=output, lse=lse, workspace=scratch)
        # CUDA Graph needs eager warmup before capture.
        for _ in range(3):
            result, result_lse = sm120_nvfp4.sparse_mla_decode(*problem, **kwargs)
        self.assertEqual(result.data_ptr(), output.data_ptr())
        self.assertEqual(result_lse.data_ptr(), lse.data_ptr())
        eager = output.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            sm120_nvfp4.sparse_mla_decode(*problem, **kwargs)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, eager, rtol=0, atol=0)
        expected, expected_lse = reference(*problem, chunks_per_cta=2)
        torch.testing.assert_close(output, expected, rtol=.04, atol=.006)
        torch.testing.assert_close(lse, expected_lse, rtol=2e-5, atol=3e-4)

    def test_binding_rejects_wrong_contract(self):
        problem = list(make_problem(batch=1))
        with self.assertRaisesRegex(RuntimeError, "dtype"):
            sm120_nvfp4.sparse_mla_decode(problem[0].half(), *problem[1:])
        with self.assertRaisesRegex(RuntimeError, "overlap"):
            sm120_nvfp4.sparse_mla_decode(*problem, output=problem[0])
        with self.assertRaisesRegex(RuntimeError, "workspace"):
            sm120_nvfp4.sparse_mla_decode(*problem, chunks_per_cta=1,
                workspace=torch.empty(1, device="cuda", dtype=torch.uint8))

    @unittest.skipUnless(os.environ.get("SPARSE_MLA_FLASHINFER_TEST") == "1",
                         "set SPARSE_MLA_FLASHINFER_TEST=1 for pinned FlashInfer integration")
    def test_flashinfer_cache_and_grouped_decode(self):
        from flashinfer.mla import nvfp4_quantize_pack_sparse_mla_cache
        from flashinfer.mla._sparse_mla_sm120._dsv4_nvfp4 import _nvfp4_sparse_mla_decode
        problem = list(make_problem(batch=1))
        torch.manual_seed(71)
        kv = (torch.randn((3, 64, 512), device="cuda") * .25).to(torch.bfloat16)
        upstream_cache = nvfp4_quantize_pack_sparse_mla_cache(kv, kv_layout="HND")
        torch.testing.assert_close(upstream_cache.reshape(3, 64, 384), pack_cache(kv), rtol=0, atol=0)
        problem[1] = upstream_cache
        problem[3].remainder_(3 * 64)
        sink = torch.linspace(-2., 2., 64, device="cuda")
        for cpb in (1, 9, 10):
            actual, lse = self.compare(problem, sink=sink, chunks_per_cta=cpb)
            expected, expected_lse = _nvfp4_sparse_mla_decode(
                problem[0], problem[1], problem[3], 512**-.5,
                extra_kv_cache=problem[2], extra_indices=problem[4],
                attn_sink=sink, chunks_per_block_override=cpb)
            torch.testing.assert_close(actual, expected, rtol=.05, atol=.008)
            torch.testing.assert_close(lse, expected_lse, rtol=2e-5, atol=3e-4)


if __name__ == "__main__":
    unittest.main()
