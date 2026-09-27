"""Run these component/state/block gates on an SM120 server, not local WSL."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch

from sm120_nvfp4 import DSV4CSAConfig, DSV4CSAAttentionBlock, sparse_mla_pack_cache
from sm120_nvfp4.dsv4 import select_csa_topk
from sm120_nvfp4.dsv4_math import (CheckpointLinear, hadamard, mxfp4_pack,
                                  mxfp4_simulate, mxfp4_unpack, pack_nvfp4)
from dsv4_block_reference import (compressor_ref, full_prefix_reference, hadamard_ref,
                                  linear_ref, mxfp4_ref, norm_ref, pack_rows_ref)
from sparse_mla_reference import prefill_reference, unpack_cache


class DSV4CSABlockTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
            raise unittest.SkipTest("SM120 required")
        cls.old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    @classmethod
    def tearDownClass(cls):
        torch.backends.cuda.matmul.allow_tf32 = cls.old_tf32

    def setUp(self):
        torch.manual_seed(1927)
        # Reduce projection ranks only. Core and indexer geometry remain the
        # actual DS-V4 shapes; full checkpoint geometry has a separate gate.
        self.config = DSV4CSAConfig(hidden_size=128, q_lora_rank=128, o_lora_rank=128)
        self.block = DSV4CSAAttentionBlock(self.config, query_chunk_size=64)
        self.block.compressor.ape.data.normal_(0, .2)
        self.block.indexer.compressor.ape.data.normal_(0, .2)
        self.block.attn_sink.data.uniform_(-2., 2.)

    def hidden(self, count):
        return torch.randn((count, self.config.hidden_size), device="cuda", dtype=torch.bfloat16)

    def assert_output(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=.06, atol=.015)
        self.assertLess((actual.float() - expected.float()).square().mean().sqrt().item(), .003)

    def assert_state(self, a, b):
        self.assertEqual(a.length, b.length)
        for name, value in a.tensors().items():
            other = b.tensors()[name]
            with self.subTest(state_tensor=name):
                if value.dtype == torch.uint8:
                    torch.testing.assert_close(value, other, rtol=0, atol=0)
                else:
                    torch.testing.assert_close(value, other, rtol=2e-5, atol=2e-5)

    def test_native_cache_pack_append_and_footer(self):
        values = torch.randn((137, 512), dtype=torch.bfloat16, device="cuda")
        actual = pack_nvfp4(values)
        torch.testing.assert_close(actual, pack_rows_ref(values), rtol=0, atol=0)
        # Append across physical page boundaries and retain all other slots.
        replacement = values[:5].mul(3).contiguous()
        slots = torch.tensor([0, 63, 64, 128, 191], device="cuda", dtype=torch.int32)
        expected_rows = torch.zeros((192, 512), dtype=torch.bfloat16, device="cuda")
        expected_rows[:137] = values
        expected_rows[slots.long()] = replacement
        sparse_mla_pack_cache(replacement, slots, actual)
        torch.testing.assert_close(actual, pack_rows_ref(expected_rows), rtol=0, atol=0)
        before = actual.clone()
        sparse_mla_pack_cache(values[:2], torch.tensor([-1, 192], device="cuda", dtype=torch.int32), actual)
        torch.testing.assert_close(actual, before, rtol=0, atol=0)
        with self.assertRaisesRegex(RuntimeError, "dtype"):
            sparse_mla_pack_cache(values.half(), torch.arange(137, device="cuda", dtype=torch.int32), actual)

    def test_indexer_rotation_mxfp4_and_ties(self):
        x = torch.randn((13, 64, 128), device="cuda", dtype=torch.bfloat16)
        torch.testing.assert_close(hadamard(x), hadamard_ref(x), rtol=.01, atol=.016)
        torch.testing.assert_close(mxfp4_simulate(x), mxfp4_ref(x), rtol=0, atol=0)
        ties = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5., -5.], device="cuda")
        values = torch.zeros((1, 128), device="cuda", dtype=torch.bfloat16)
        values[0, :8] = ties
        values[0, 31] = 6.  # exactly unit power-of-two scale in this group
        rounded = mxfp4_simulate(values)
        expected = torch.tensor([0., 1., 1., 2., 2., 4., 4., -4.], device="cuda", dtype=torch.bfloat16)
        torch.testing.assert_close(rounded[0, :8], expected, rtol=0, atol=0)

    def test_overlapping_compressor_and_partial_tail(self):
        x = self.hidden(139)
        for compressor, attr in ((self.block.compressor, "attention_compressor"),
                                 (self.block.indexer.compressor, "index_compressor")):
            state = self.block.new_state(200)
            reference, values, scores = compressor_ref(x, compressor, self.block.frequencies)
            pieces = []
            for begin, end in ((0, 1), (1, 3), (3, 4), (4, 9), (9, 130), (130, 139)):
                pieces.append(compressor.append(x[begin:end], getattr(state, attr), begin, self.block.frequencies))
            torch.testing.assert_close(torch.cat(pieces), reference, rtol=.01, atol=.016)
            tail = getattr(state, attr)
            torch.testing.assert_close(tail.previous_values, values[132:136, :compressor.dim], rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(tail.previous_scores, scores[132:136, :compressor.dim], rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(tail.pending_values, values[136:], rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(tail.pending_scores, scores[136:], rtol=2e-5, atol=2e-5)

    def test_topk_above_512_visibility_and_key_tiling(self):
        state = self.block.new_state(4096)
        key = torch.randn((777, 128), dtype=torch.bfloat16, device="cuda")
        state.index_payload[:777], state.index_scales[:777] = mxfp4_pack(key)
        q = mxfp4_simulate(torch.randn((5, 64, 128), dtype=torch.bfloat16, device="cuda"))
        weights = torch.randn((5, 64), dtype=torch.bfloat16, device="cuda") * .02
        positions = torch.tensor([0, 3, 2047, 2051, 3107], device="cuda")
        a, lengths = select_csa_topk(q, weights, state, positions, 777, key_tile=256)
        b, _ = select_csa_topk(q, weights, state, positions, 777, key_tile=73, query_tile=2)
        keys = mxfp4_unpack(state.index_payload[:777], state.index_scales[:777])
        scores = (torch.einsum("qhd,kd->qhk", q, keys).relu() * weights[..., None]).sum(1).float()
        visible = (positions + 1) // 4
        scores.masked_fill_(torch.arange(777, device="cuda")[None] >= visible[:, None], -torch.inf)
        selected = scores.argsort(-1, descending=True, stable=True)[:, :512]
        expected = torch.where(selected < visible[:, None], selected, 2147483647).sort(-1).values
        expected = torch.where(expected == 2147483647, -1, expected).int()
        torch.testing.assert_close(a, expected, rtol=0, atol=0)
        torch.testing.assert_close(b, expected, rtol=0, atol=0)
        torch.testing.assert_close(lengths, visible.clamp_max(512).int(), rtol=0, atol=0)
        tied, _ = select_csa_topk(q, torch.zeros_like(weights), state, positions, 777)
        torch.testing.assert_close(tied[-1], torch.arange(512, device="cuda", dtype=torch.int32), rtol=0, atol=0)

    def test_full_block_against_stateless_reference(self):
        hidden = self.hidden(137)
        expected, details = full_prefix_reference(self.block, hidden)
        state, trace = self.block.new_state(200), []
        actual = self.block.prefill(hidden, [state], [137], trace=trace)
        self.assert_output(actual, expected)
        ids = torch.cat([item["compressed_indices"] for item in trace])
        torch.testing.assert_close(ids, details["indices"], rtol=0, atol=0)
        torch.testing.assert_close(torch.cat([item["compressed_lengths"] for item in trace]), details["compressed_lengths"], rtol=0, atol=0)
        # Inspect storage from the final state against a fresh prefix rebuild.
        stored = unpack_cache(state.compressed_cache)[:state.compressed_length]
        expected_cache = unpack_cache(pack_rows_ref(details["compressed"]))[:state.compressed_length]
        torch.testing.assert_close(stored, expected_cache, rtol=.02, atol=.032)
        ring = unpack_cache(state.swa_cache)
        kv = unpack_cache(pack_rows_ref(details["kv"]))[:137]
        logical = torch.arange(9, 137, device="cuda")
        torch.testing.assert_close(ring[logical % 128], kv[logical], rtol=.02, atol=.032)

    def test_prefill_chunks_then_decode_state_and_outputs(self):
        hidden = self.hidden(141)
        whole_state, split_state = self.block.new_state(200), self.block.new_state(200)
        whole = self.block.prefill(hidden, [whole_state], [141])
        pieces = []
        for begin, end in ((0, 3), (3, 67), (67, 129), (129, 133)):
            pieces.append(self.block.prefill(hidden[begin:end], [split_state], [end - begin]))
        for row in range(133, 141):
            pieces.append(self.block.decode(hidden[row:row + 1], [split_state]))
        self.assert_output(torch.cat(pieces), whole)
        self.assert_state(split_state, whole_state)

    def test_future_tokens_cannot_change_prefix(self):
        hidden = self.hidden(137)
        changed = hidden.clone()
        changed[67:].neg_().mul_(3)
        a = self.block.prefill(hidden, [self.block.new_state(200)], [137])
        b = self.block.prefill(changed, [self.block.new_state(200)], [137])
        torch.testing.assert_close(a[:67], b[:67], rtol=0, atol=0)

    def test_ragged_requests_fork_reset_and_validation(self):
        x, y = self.hidden(135), self.hidden(9)
        a, b, empty = [self.block.new_state(200) for _ in range(3)]
        actual = self.block.prefill(torch.cat((x, y)), [a, empty, b], [135, 0, 9])
        aa, bb = self.block.new_state(200), self.block.new_state(200)
        expected = torch.cat((self.block.prefill(x, [aa], [135]), self.block.prefill(y, [bb], [9])))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assert_state(a, aa)
        self.assertEqual(empty.length, 0)
        child, sibling = a.fork(), a.fork()
        for name in a.tensors():
            if a.tensors()[name].numel():
                self.assertNotEqual(a.tensors()[name].data_ptr(), child.tensors()[name].data_ptr())
        tail = self.hidden(5)
        first = self.block.prefill(tail, [child], [5])
        second = torch.cat([self.block.decode(tail[i:i + 1], [sibling]) for i in range(5)])
        self.assert_output(first, second)
        self.assert_state(child, sibling)
        self.assert_state(a, aa)
        child.reset()
        self.assert_state(child, self.block.new_state(200))
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.block.prefill(self.hidden(2), [a, a], [1, 1])
        with self.assertRaisesRegex(ValueError, "capacity"):
            self.block.prefill(self.hidden(201), [child], [201])
        self.assertEqual(child.length, 0)
        def fail(*args, **kwargs):
            raise RuntimeError("injected core failure")
        with self.assertRaisesRegex(RuntimeError, "injected"):
            self.block.decode(self.hidden(1), [child], backend=fail)
        with self.assertRaisesRegex(ValueError, "failed"):
            self.block.decode(self.hidden(1), [child])
        child.reset()
        self.block.decode(self.hidden(1), [child])

    def test_mhc_attention_branch(self):
        c = self.config
        self.block.hc_attn_fn.data.normal_(0, .01)
        self.block.hc_attn_base.data.normal_(0, .1)
        x = torch.randn((5, c.hc_mult, c.hidden_size), device="cuda", dtype=torch.bfloat16)
        flat = x.flatten(1).float()
        mixed = torch.nn.functional.linear(flat, self.block.hc_attn_fn) / torch.sqrt(flat.square().mean(-1, keepdim=True) + c.rms_norm_eps)
        pre = (mixed[:, :4] * self.block.hc_attn_scale[0] + self.block.hc_attn_base[:4]).sigmoid() + c.hc_eps
        post = 2 * (mixed[:, 4:8] * self.block.hc_attn_scale[1] + self.block.hc_attn_base[4:8]).sigmoid()
        matrix = (mixed[:, 8:] * self.block.hc_attn_scale[2] + self.block.hc_attn_base[8:]).reshape(-1, 4, 4).softmax(-1) + c.hc_eps
        matrix /= matrix.sum(-2, keepdim=True) + c.hc_eps
        for _ in range(c.hc_sinkhorn_iters - 1):
            matrix /= matrix.sum(-1, keepdim=True) + c.hc_eps
            matrix /= matrix.sum(-2, keepdim=True) + c.hc_eps
        collapsed = (pre[..., None] * x.float()).sum(1).to(torch.bfloat16)
        expected_core, _ = full_prefix_reference(self.block, collapsed)
        expected = (post[..., None] * expected_core.float()[:, None] +
                    torch.einsum("tij,tid->tjd", matrix, x.float())).to(torch.bfloat16)
        actual = self.block.forward_hc(x, [self.block.new_state(32)], [5])
        self.assert_output(actual, expected)

    def test_fp8_projection_contract(self):
        projection = CheckpointLinear(256, 256, device="cuda")
        weight = torch.randn((256, 256), device="cuda").to(torch.float8_e4m3fn)
        scale = torch.tensor([[.25, 2.], [4., .5]], device="cuda")
        projection.load_weight(weight, scale)
        x = torch.randn((7, 256), device="cuda", dtype=torch.bfloat16)
        torch.testing.assert_close(projection(x), linear_ref(x, projection), rtol=.01, atol=.03125)

    def test_full_flash_geometry(self):
        block = DSV4CSAAttentionBlock(DSV4CSAConfig(), query_chunk_size=8)
        hidden = torch.randn((9, 4096), device="cuda", dtype=torch.bfloat16)
        state = block.new_state(16)
        output = block.prefill(hidden, [state], [9])
        expected, _ = full_prefix_reference(block, hidden)
        self.assertEqual(output.shape, (9, 4096))
        self.assert_output(output, expected)

    def test_official_checkpoint_key_and_scale_loading(self):
        from safetensors.torch import save_file
        prefix = "layers.2."
        tensors = {}
        # Exercise real key names across shards, including FP8 wo_a -> BF16
        # conversion. Values are synthetic; this is not a pretrained fixture.
        for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            module = getattr(self.block, name)
            tensors[prefix + "attn." + name + ".weight"] = module.weight.detach().cpu().float().to(torch.float8_e4m3fn)
            tensors[prefix + "attn." + name + ".scale"] = torch.ones(
                ((module.out_features + 127) // 128, (module.in_features + 127) // 128)) * .5
        for name, value in (("attn_norm.weight", self.block.attn_norm),
                            ("attn.q_norm.weight", self.block.q_norm),
                            ("attn.kv_norm.weight", self.block.kv_norm),
                            ("attn.attn_sink", self.block.attn_sink)):
            tensors[prefix + name] = value.detach().cpu()
        for name in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale"):
            tensors[prefix + name] = getattr(self.block, name).detach().cpu()
        for name in ("wq_b", "weights_proj"):
            tensors[prefix + "attn.indexer." + name + ".weight"] = getattr(self.block.indexer, name).weight.detach().cpu()
        for module, name in ((self.block.compressor, "attn.compressor"),
                             (self.block.indexer.compressor, "attn.indexer.compressor")):
            for part in ("wkv", "wgate"):
                tensors[prefix + name + "." + part + ".weight"] = getattr(module, part).weight.detach().cpu().bfloat16()
            tensors[prefix + name + ".ape"] = module.ape.detach().cpu()
            tensors[prefix + name + ".norm.weight"] = module.norm_weight.detach().cpu()
        config = dict(vars(self.config), model_type="deepseek_v4", num_key_value_heads=1,
            compress_ratios=[0, 0, 4], rope_scaling=dict(type="yarn", factor=16,
            original_max_position_embeddings=65536, beta_fast=32, beta_slow=1))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath("config.json").write_text(json.dumps(config))
            shards = ({}, {})
            mapping = {}
            for i, (name, value) in enumerate(tensors.items()):
                shards[i % 2][name] = value.contiguous()
                mapping[name] = f"shard{i % 2}.safetensors"
            for i, shard in enumerate(shards):
                save_file(shard, str(root / f"shard{i}.safetensors"))
            root.joinpath("model.safetensors.index.json").write_text(json.dumps(dict(weight_map=mapping)))
            snapshot = root / "snapshot"
            snapshot.mkdir()
            for name in ("config.json", "model.safetensors.index.json", "shard0.safetensors", "shard1.safetensors"):
                (snapshot / name).symlink_to(root / name)
            loaded = DSV4CSAAttentionBlock.from_checkpoint(snapshot, layer_id=2, query_chunk_size=8)
            self.assertEqual(loaded.wq_a.weight.dtype, torch.float8_e4m3fn)
            self.assertEqual(loaded.wo_a.weight.dtype, torch.bfloat16)
            expected_wo = (tensors[prefix + "attn.wo_a.weight"].float() * .5).bfloat16().cuda()
            torch.testing.assert_close(loaded.wo_a.weight, expected_wo, rtol=0, atol=0)
            hidden = self.hidden(9)
            actual = loaded.prefill(hidden, [loaded.new_state(16)], [9])
            expected, _ = full_prefix_reference(loaded, hidden)
            self.assert_output(actual, expected)
            with self.assertRaisesRegex(ValueError, "CSA"):
                DSV4CSAAttentionBlock.from_checkpoint(root, layer_id=1)

    @unittest.skipUnless(os.environ.get("DSV4_CHECKPOINT"), "set DSV4_CHECKPOINT to an official local snapshot")
    def test_real_checkpoint_loader_and_block(self):
        block = DSV4CSAAttentionBlock.from_checkpoint(os.environ["DSV4_CHECKPOINT"], layer_id=2, query_chunk_size=8)
        hidden = torch.randn((9, 4096), device="cuda", dtype=torch.bfloat16)
        actual = block.prefill(hidden, [block.new_state(32)], [9])
        expected, _ = full_prefix_reference(block, hidden)
        self.assert_output(actual, expected)

    @unittest.skipUnless(os.environ.get("SPARSE_MLA_FLASHINFER_TEST") == "1", "optional pinned FlashInfer block A/B")
    def test_flashinfer_block_backend(self):
        hidden = self.hidden(17)
        native_state, baseline_state = self.block.new_state(32), self.block.new_state(32)
        native = self.block.prefill(hidden, [native_state], [17])
        baseline = self.block.prefill(hidden, [baseline_state], [17], backend="flashinfer")
        self.assert_output(native, baseline)
        self.assert_state(native_state, baseline_state)
        tail = self.hidden(1)
        native = self.block.decode(tail, [native_state])
        baseline = self.block.decode(tail, [baseline_state], backend="flashinfer")
        self.assert_output(native, baseline)


if __name__ == "__main__":
    unittest.main()
