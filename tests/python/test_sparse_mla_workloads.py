"""CSA/HCA core contracts, without implementing a model or compressor."""
import unittest

import torch
import sm120_nvfp4

from attention_workloads import attention_workload
from sparse_mla_reference import prefill_reference, reference


class SparseMlaWorkloadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
            raise unittest.SkipTest("SM120 required")
        cls.old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    @classmethod
    def tearDownClass(cls):
        torch.backends.cuda.matmul.allow_tf32 = cls.old_tf32

    def test_causal_csa_hca_boundaries_and_request_isolation(self):
        for kind, ratio in (("csa", 4), ("hca", 128)):
            for history in (ratio - 1, ratio, ratio * 65 - 1, ratio * 513 - 1):
                with self.subTest(kind=kind, history=history):
                    problem, kwargs, meta = attention_workload(2, 3, history, kind,
                        mixed_lengths=True, seed=1927)
                    for row, position in enumerate(meta["query_positions"]):
                        request = row // 3
                        expected_count = (position + 1) // ratio
                        if kind == "csa":
                            expected_count = min(expected_count, 512)
                        self.assertEqual(meta["selected_compressed_rows_per_query"][row], expected_count)
                        swa_slots = problem[3][row, :meta["swa_rows_per_query"][row]].long()
                        base = max(0, meta["history_tokens"][request] - 127)
                        logical = swa_slots - request * meta["swa_slots_per_request"] + base
                        torch.testing.assert_close(logical,
                            torch.arange(max(0, position - 127), position + 1, device="cuda"))
                        comp_slots = problem[4][row, :expected_count].long()
                        ids = comp_slots - request * meta["compressed_slots_per_request"]
                        self.assertTrue(((ids >= 0) & (ids < (position + 1) // ratio)).all().item())
                        self.assertEqual(ids.unique().numel(), expected_count)
                        if kind == "hca":
                            torch.testing.assert_close(ids, torch.arange(expected_count, device="cuda"))
                    expected, expected_lse = prefill_reference(*problem, **kwargs)
                    out, lse = sm120_nvfp4.sparse_mla_prefill(*problem, **kwargs)
                    torch.testing.assert_close(out, expected, rtol=.05, atol=.02)
                    self.assertLess((out.float() - expected.float()).square().mean().sqrt().item(), .005)
                    torch.testing.assert_close(lse, expected_lse, rtol=2e-5, atol=5e-4)
                    direct, direct_lse = sm120_nvfp4.sparse_mla_decode(*problem, **kwargs, chunks_per_cta=2147483647)
                    torch.testing.assert_close(out, direct, rtol=0, atol=0)
                    torch.testing.assert_close(lse, direct_lse, rtol=0, atol=0)
                    if history == ratio * 65 - 1:
                        split, split_lse = sm120_nvfp4.sparse_mla_decode(*problem, **kwargs, chunks_per_cta=1)
                        split_ref, split_ref_lse = reference(*problem, **kwargs, chunks_per_cta=1)
                        torch.testing.assert_close(split, split_ref, rtol=.05, atol=.02)
                        self.assertLess((split.float() - split_ref.float()).square().mean().sqrt().item(), .005)
                        torch.testing.assert_close(split_lse, split_ref_lse, rtol=2e-5, atol=5e-4)


if __name__ == "__main__":
    unittest.main()
