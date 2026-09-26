"""Fixed-index sparse prefill gates; execute only on the SM120 GPU server."""
import math
import os
import unittest

import torch
import sm120_nvfp4
from sparse_mla_reference import make_prefill_problem, pack_cache, prefill_reference


class SparseMlaPrefillTest(unittest.TestCase):
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
        expected, expected_lse = prefill_reference(*problem, **kwargs)
        actual, lse = sm120_nvfp4.sparse_mla_prefill(*problem, **kwargs)
        torch.cuda.synchronize()
        self.assertEqual(actual.shape, problem[0].shape)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertEqual(lse.shape, (problem[0].shape[0], 64))
        torch.testing.assert_close(actual, expected, rtol=.04, atol=.006)
        self.assertLess((actual.float() - expected.float()).square().mean().sqrt().item(), .0015)
        torch.testing.assert_close(lse, expected_lse, rtol=2e-5, atol=3e-4)
        return actual, lse

    def test_csa_full_lists_and_direct_decode(self):
        # Cover both sides of decode's B=64 schedule threshold. Prefill must
        # always use the all-chunk path, with no BF16 split-output rounding.
        for queries in (1, 7, 65):
            with self.subTest(queries=queries):
                problem = make_prefill_problem(queries=queries)
                sink = torch.linspace(-2., 4., 64, device="cuda")
                actual, lse = self.compare(problem, sink=sink)
                direct, direct_lse = sm120_nvfp4.sparse_mla_decode(
                    *problem, sink=sink, chunks_per_cta=10)
                torch.testing.assert_close(actual, direct, rtol=0, atol=0)
                torch.testing.assert_close(lse, direct_lse, rtol=0, atol=0)

    def test_per_query_lengths_tails_and_masks(self):
        problem = list(make_prefill_problem(queries=7, swa=71, compressed=137))
        problem[1] = problem[1][:, None]  # HND packed cache
        problem[2] = problem[2][:, :, None]  # NHD packed cache
        problem[3][:, 2] = -1
        problem[3][:, 3] = problem[3][:, 1]  # repeated occurrence
        problem[4][:, 7] = problem[2].shape[0] * 64  # out of range
        swa_lengths = torch.tensor([0, 1, 63, 64, 71, 99, -2], device="cuda", dtype=torch.int32)
        comp_lengths = torch.tensor([0, 1, 17, 65, 129, 137, 200], device="cuda", dtype=torch.int32)
        sink = torch.linspace(-2., 2., 64, device="cuda")
        sink[::2] = -torch.inf
        self.compare(problem, swa_lengths=swa_lengths, compressed_lengths=comp_lengths,
                     sink=sink, softmax_scale=.073, lse_scale=math.log(2))

    def test_query_chunks_and_ragged_requests(self):
        q, c1, c2, i1, i2 = make_prefill_problem(queries=70, swa=128, compressed=512)
        # Packed request boundaries [0,3,3,68,70], including an empty request.
        # Disjoint slot ranges encode request isolation; the core only sees IDs.
        for begin, end, slot_begin, slot_end in ((0, 3, 0, 32), (3, 68, 32, 96), (68, 70, 96, 128)):
            i1[begin:end].remainder_(slot_end - slot_begin).add_(slot_begin)
            i2[begin:end].remainder_(slot_end - slot_begin).add_(slot_begin)
        sl = torch.arange(70, device="cuda", dtype=torch.int32).remainder(129)
        cl = (torch.arange(70, device="cuda", dtype=torch.int32) * 17).remainder(513)
        sink = torch.linspace(-3., 3., 64, device="cuda")
        whole, whole_lse = self.compare((q, c1, c2, i1, i2), swa_lengths=sl,
                                        compressed_lengths=cl, sink=sink)
        # Slice the query axis only; every query retains its candidate order
        # and capacity. Chunk boundaries deliberately cross request boundaries.
        for boundaries in ((0, 1, 34, 65, 70), (0, 3, 68, 70)):
            outputs, lses = [], []
            for begin, end in zip(boundaries, boundaries[1:]):
                out, lse = sm120_nvfp4.sparse_mla_prefill(
                    q[begin:end], c1, c2, i1[begin:end], i2[begin:end],
                    swa_lengths=sl[begin:end], compressed_lengths=cl[begin:end], sink=sink)
                outputs.append(out)
                lses.append(lse)
            torch.testing.assert_close(torch.cat(outputs), whole, rtol=0, atol=0)
            torch.testing.assert_close(torch.cat(lses), whole_lse, rtol=0, atol=0)
        # A token-at-a-time direct decode consumes exactly the same lists.
        for row in (0, 2, 3, 33, 67, 69):
            selection = slice(row, row + 1)
            out, lse = sm120_nvfp4.sparse_mla_decode(
                q[selection], c1, c2, i1[selection], i2[selection],
                swa_lengths=sl[selection], compressed_lengths=cl[selection],
                sink=sink, chunks_per_cta=10)
            torch.testing.assert_close(out, whole[selection], rtol=0, atol=0)
            torch.testing.assert_close(lse, whole_lse[selection], rtol=0, atol=0)

    def test_unselected_cache_rows_do_not_contribute(self):
        q, c1, c2, i1, i2 = make_prefill_problem(queries=3, swa=71, compressed=137)
        i1.remainder_(64)
        i2.remainder_(64)
        # The second page appears only in each list's masked suffix. Its
        # contents must not affect either V requantization or the denominator.
        i1[:, 17:] = 64
        i2[:, 67:] = 64
        kwargs = dict(swa_lengths=torch.full((3,), 17, dtype=torch.int32, device="cuda"),
                      compressed_lengths=torch.full((3,), 67, dtype=torch.int32, device="cuda"))
        expected, expected_lse = self.compare((q, c1, c2, i1, i2), **kwargs)
        poison = pack_cache(torch.full((1, 64, 512), 32., dtype=torch.bfloat16, device="cuda"))
        c1[1:2].copy_(poison)
        c2[1:2].copy_(poison)
        actual, lse = sm120_nvfp4.sparse_mla_prefill(q, c1, c2, i1, i2, **kwargs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(lse, expected_lse, rtol=0, atol=0)

    def test_empty_candidates_and_sink(self):
        for swa, compressed in ((0, 0), (0, 73), (67, 0), (67, 73)):
            with self.subTest(swa=swa, compressed=compressed):
                problem = list(make_prefill_problem(queries=3, swa=swa, compressed=compressed))
                if swa == 0:
                    problem[1] = torch.empty((0, 64, 384), dtype=torch.uint8, device="cuda")
                if compressed == 0:
                    problem[2] = torch.empty((0, 64, 384), dtype=torch.uint8, device="cuda")
                # First check a single populated pool, then fully masked lists.
                self.compare(problem)
                problem[3].fill_(-1)
                problem[4].fill_(-1)
                for sink in (None, torch.linspace(-3., 3., 64, device="cuda"),
                             torch.full((64,), -torch.inf, device="cuda")):
                    out, lse = self.compare(problem, sink=sink)
                    torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)
                    expected = (torch.full_like(lse, -torch.inf) if sink is None else
                                sink[None].expand_as(lse) * math.log2(math.e))
                    torch.testing.assert_close(lse, expected, rtol=2e-6, atol=2e-6)

    def test_fp32_denominator_and_bf16_rope(self):
        q, _, c2, i1, i2 = make_prefill_problem(queries=3, swa=64, compressed=0)
        q.zero_()
        c1 = pack_cache(torch.ones((1, 64, 512), dtype=torch.bfloat16, device="cuda"))
        i1[:] = torch.arange(64, device="cuda", dtype=torch.int32)
        out, lse = self.compare((q, c1, c2, i1, i2))
        torch.testing.assert_close(out[..., :448], torch.full_like(out[..., :448], 1.0625), rtol=0, atol=0)
        torch.testing.assert_close(out[..., 448:], torch.ones_like(out[..., 448:]), rtol=0, atol=0)
        torch.testing.assert_close(lse, torch.full_like(lse, 6.), rtol=0, atol=0)

    def test_reusable_buffers_graph_and_changed_inputs(self):
        problem = make_prefill_problem(queries=3, swa=71, compressed=137)
        output = torch.empty_like(problem[0])
        lse = torch.empty((3, 64), dtype=torch.float32, device="cuda")
        lengths = torch.full((3,), 71, dtype=torch.int32, device="cuda")
        kwargs = dict(output=output, lse=lse, swa_lengths=lengths)
        for _ in range(3):
            actual, actual_lse = sm120_nvfp4.sparse_mla_prefill(*problem, **kwargs)
        self.assertEqual(actual.data_ptr(), output.data_ptr())
        self.assertEqual(actual_lse.data_ptr(), lse.data_ptr())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            sm120_nvfp4.sparse_mla_prefill(*problem, **kwargs)
        problem[0].mul_(.5)
        problem[3][:, 0] = 0
        lengths[:] = torch.tensor([0, 17, 65], dtype=torch.int32, device="cuda")
        expected, expected_lse = prefill_reference(*problem, swa_lengths=lengths)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, rtol=.04, atol=.006)
        torch.testing.assert_close(lse, expected_lse, rtol=2e-5, atol=3e-4)

    def test_binding_rejects_wrong_contract(self):
        q, c1, c2, i1, i2 = make_prefill_problem(queries=3)
        for bad in (q[:, None], q[:, :32].contiguous(), q[:0]):
            with self.subTest(shape=bad.shape), self.assertRaises(RuntimeError):
                sm120_nvfp4.sparse_mla_prefill(bad, c1, c2, i1, i2)
        with self.assertRaisesRegex(RuntimeError, "dtype"):
            sm120_nvfp4.sparse_mla_prefill(q.half(), c1, c2, i1, i2)
        for bad in (i1[:, None], i1[:1]):
            with self.assertRaisesRegex(RuntimeError, "shape"):
                sm120_nvfp4.sparse_mla_prefill(q, c1, c2, bad, i2)
        with self.assertRaisesRegex(RuntimeError, "overlap"):
            sm120_nvfp4.sparse_mla_prefill(q, c1, c2, i1, i2, output=q)
        with self.assertRaisesRegex(RuntimeError, "shape"):
            sm120_nvfp4.sparse_mla_prefill(q, c1, c2, i1, i2,
                swa_lengths=torch.ones(1, dtype=torch.int32, device="cuda"))
        with self.assertRaisesRegex(RuntimeError, "positive"):
            sm120_nvfp4.sparse_mla_prefill(q, c1, c2, i1, i2, softmax_scale=0.)
        with self.assertRaisesRegex(RuntimeError, "lse"):
            sm120_nvfp4.sparse_mla_prefill(q, c1, c2, i1, i2,
                lse=torch.empty((3, 32), dtype=torch.float32, device="cuda"))

    @unittest.skipUnless(os.environ.get("SPARSE_MLA_FLASHINFER_TEST") == "1",
                         "set SPARSE_MLA_FLASHINFER_TEST=1 for pinned FlashInfer integration")
    def test_flashinfer_streaming_prefill(self):
        from flashinfer.mla._sparse_mla_sm120._dsv4_nvfp4 import _nvfp4_sparse_mla_prefill
        problem = make_prefill_problem(queries=7)
        sl = torch.tensor([0, 1, 17, 63, 64, 127, 128], dtype=torch.int32, device="cuda")
        cl = torch.tensor([0, 0, 1, 65, 137, 511, 512], dtype=torch.int32, device="cuda")
        sink = torch.linspace(-2., 3., 64, device="cuda")
        actual, lse = self.compare(problem, swa_lengths=sl, compressed_lengths=cl,
                                   sink=sink, lse_scale=math.log(2))
        expected, expected_lse = _nvfp4_sparse_mla_prefill(
            problem[0], problem[1], problem[3], 512**-.5, topk_length=sl,
            extra_kv_cache=problem[2], extra_indices=problem[4],
            extra_topk_length=cl, attn_sink=sink, lse_scale=math.log(2))
        torch.testing.assert_close(actual, expected, rtol=.05, atol=.008)
        torch.testing.assert_close(lse, expected_lse, rtol=2e-5, atol=3e-4)


if __name__ == "__main__":
    unittest.main()
