"""Standalone cache operator gates; no checkpoint/block runtime required."""
import os
import unittest

import torch
import sm120_nvfp4

from sparse_mla_reference import pack_cache


class SparseMlaCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
            raise unittest.SkipTest("SM120 required")

    def setUp(self):
        torch.manual_seed(713)

    def test_pack_page_tails_zero_scale_and_rounding(self):
        for rows in (0, 1, 63, 64, 65, 137):
            with self.subTest(rows=rows):
                padded = ((rows + 63) // 64) * 64
                values = (torch.randn((padded, 512), device="cuda") * .25).bfloat16()
                if rows:
                    values[rows:].zero_()
                    values[0, :16] = 1.e-7  # scale rounds to zero
                    values[0, 16:32] = 0
                    # Choose amax=6 so the scale is exactly one; RNE ties.
                    values[0, 32:40] = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5., 6.], device="cuda")
                cache = torch.zeros((padded // 64, 64, 384), device="cuda", dtype=torch.uint8)
                result = sm120_nvfp4.sparse_mla_pack_cache(values[:rows],
                    torch.arange(rows, device="cuda", dtype=torch.int32), cache)
                self.assertEqual(result.data_ptr(), cache.data_ptr())
                torch.testing.assert_close(cache, pack_cache(values.reshape(-1, 64, 512)), rtol=0, atol=0)

    def test_scattered_append_preserves_unwritten_rows(self):
        latent = (torch.randn((192, 512), device="cuda") * .25).bfloat16()
        cache = pack_cache(latent.reshape(3, 64, 512))
        values = torch.randn((7, 512), device="cuda", dtype=torch.bfloat16)
        slots = torch.tensor([191, 0, 64, 63, 128, -1, 192], device="cuda", dtype=torch.int32)
        expected = latent.clone()
        expected[slots[:5].long()] = values[:5]
        sm120_nvfp4.sparse_mla_pack_cache(values, slots, cache)
        torch.testing.assert_close(cache, pack_cache(expected.reshape(3, 64, 512)), rtol=0, atol=0)

    def test_graph_replay_observes_new_values_and_slots(self):
        values = torch.randn((3, 512), device="cuda", dtype=torch.bfloat16)
        slots = torch.tensor([0, 63, 64], device="cuda", dtype=torch.int32)
        cache = torch.zeros((3, 64, 384), device="cuda", dtype=torch.uint8)
        sm120_nvfp4.sparse_mla_pack_cache(values, slots, cache)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            sm120_nvfp4.sparse_mla_pack_cache(values, slots, cache)
        expected = torch.zeros((192, 512), device="cuda", dtype=torch.bfloat16)
        expected[slots.long()] = values
        slots.copy_(torch.tensor([127, 128, 191], device="cuda", dtype=torch.int32))
        values.mul_(2)
        expected[slots.long()] = values
        graph.replay()
        torch.testing.assert_close(cache, pack_cache(expected.reshape(3, 64, 512)), rtol=0, atol=0)

    def test_binding_validation_and_layout_aliases(self):
        values = torch.randn((2, 512), device="cuda", dtype=torch.bfloat16)
        slots = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        cache = torch.zeros((2, 64, 384), device="cuda", dtype=torch.uint8)
        for view in (cache, cache[:, None], cache[:, :, None]):
            sm120_nvfp4.sparse_mla_pack_cache(values, slots, view)
        for invalid in (values.half(), values[:, ::2], values[:, None]):
            with self.assertRaises(RuntimeError):
                sm120_nvfp4.sparse_mla_pack_cache(invalid, slots, cache)
        with self.assertRaises(RuntimeError):
            sm120_nvfp4.sparse_mla_pack_cache(values, slots.long(), cache)
        with self.assertRaises(RuntimeError):
            sm120_nvfp4.sparse_mla_pack_cache(values, slots[:1], cache)
        with self.assertRaisesRegex(RuntimeError, "overlap"):
            overlapping = cache.flatten()[:2048].view(torch.bfloat16).reshape(2, 512)
            sm120_nvfp4.sparse_mla_pack_cache(overlapping, slots, cache)

    @unittest.skipUnless(os.environ.get("SPARSE_MLA_FLASHINFER_TEST") == "1", "optional pinned FlashInfer cache ABI")
    def test_flashinfer_append_exact_bytes(self):
        from flashinfer.mla import nvfp4_quantize_append_sparse_mla_cache
        values = torch.randn((65, 512), device="cuda", dtype=torch.bfloat16)
        slots = torch.randperm(192, device="cuda")[:65].int()
        actual = torch.zeros((3, 64, 384), device="cuda", dtype=torch.uint8)
        baseline = actual.clone()
        sm120_nvfp4.sparse_mla_pack_cache(values, slots, actual)
        nvfp4_quantize_append_sparse_mla_cache(values, slots, baseline)
        torch.testing.assert_close(actual, baseline, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
