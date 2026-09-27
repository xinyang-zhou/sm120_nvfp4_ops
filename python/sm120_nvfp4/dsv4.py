"""Stateful DeepSeek-V4-Flash CSA attention branch on SM120.

The attention core is native C++ CuTe NVFP4. Surrounding model operations use
PyTorch GPU tensors. See docs/DSV4_CSA_BLOCK.md for precision and scope.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from .dsv4_math import (OFFICIAL_REVISION, CheckpointLinear, append_nvfp4,
                        copy_nvfp4_rows, dequantize_weight, hadamard, hc_post,
                        hc_pre, mxfp4_pack, mxfp4_simulate, mxfp4_unpack,
                        pack_nvfp4, rms_norm, rope_frequencies, rotary)
from .dsv4_state import DSV4CSAConfig, DSV4CSAState


class CSACompressor(nn.Module):
    def __init__(self, config, dim, *, device):
        super().__init__()
        self.dim, self.eps = dim, config.rms_norm_eps
        self.wkv = CheckpointLinear(config.hidden_size, 2 * dim, device=device, dtype=torch.float32)
        self.wgate = CheckpointLinear(config.hidden_size, 2 * dim, device=device, dtype=torch.float32)
        self.ape = nn.Parameter(torch.zeros((4, 2 * dim), device=device, dtype=torch.float32), requires_grad=False)
        self.norm_weight = nn.Parameter(torch.ones(dim, device=device, dtype=torch.float32), requires_grad=False)

    def append(self, x, state, start, frequencies):
        """Commit complete overlapping blocks; retain previous block + tail."""
        if state.pending_values.shape[0] != start % 4:
            raise ValueError("compressor tail does not match request position")
        positions = torch.arange(start, start + x.shape[0], device=x.device)
        values = torch.cat((state.pending_values, self.wkv(x.float())), 0)
        scores = torch.cat((state.pending_scores, self.wgate(x.float()) + self.ape[positions % 4]), 0)
        complete = values.shape[0] // 4
        cutoff = complete * 4
        block_start = start - state.pending_values.shape[0]
        if complete:
            v = values[:cutoff].reshape(complete, 4, 2 * self.dim)
            s = scores[:cutoff].reshape(complete, 4, 2 * self.dim)
            previous_v = torch.cat((state.previous_values[None], v[:-1, :, :self.dim]), 0)
            previous_s = torch.cat((state.previous_scores[None], s[:-1, :, :self.dim]), 0)
            windows_v = torch.cat((previous_v, v[:, :, self.dim:]), 1)
            windows_s = torch.cat((previous_s, s[:, :, self.dim:]), 1)
            pooled = (windows_v * windows_s.softmax(1)).sum(1).to(torch.bfloat16)
            pooled = rms_norm(pooled, self.norm_weight, self.eps)
            starts = torch.arange(complete, device=x.device) * 4 + block_start
            pooled = rotary(pooled, starts, frequencies)
            state.previous_values = v[-1, :, :self.dim].clone()
            state.previous_scores = s[-1, :, :self.dim].clone()
        else:
            pooled = x.new_empty((0, self.dim))
        # Clone, rather than holding views into an arbitrarily long prompt.
        state.pending_values = values[cutoff:].clone()
        state.pending_scores = scores[cutoff:].clone()
        return pooled


def select_csa_topk(query, weights, state, positions, available, *, key_tile=256, query_tile=32):
    """GPU selection with bounded score tiles and stable tie policy.

    Rank by score descending, then logical slot ascending on exact ties.
    Return the selected set in chronological order so native NVFP4 candidate
    groups do not depend on the ordering chosen by torch.topk for tied scores.
    """
    count = query.shape[0]
    result = torch.full((count, 512), -1, dtype=torch.int32, device=query.device)
    lengths = torch.minimum((positions + 1) // 4, positions.new_full((), 512)).to(torch.int32)
    if not available:
        return result, lengths
    for qb in range(0, count, query_tile):
        q = query[qb:qb + query_tile]
        w = weights[qb:qb + query_tile]
        visible = (positions[qb:qb + query_tile] + 1) // 4
        best_scores = torch.empty((q.shape[0], 0), device=q.device, dtype=torch.float32)
        best_ids = torch.empty((q.shape[0], 0), device=q.device, dtype=torch.int64)
        for kb in range(0, available, key_tile):
            end = min(kb + key_tile, available)
            keys = mxfp4_unpack(state.index_payload[kb:end], state.index_scales[kb:end])
            # Match the official BF16 score, ReLU, weighted-head sum order.
            scores = (torch.einsum("qhd,kd->qhk", q, keys).relu() * w[..., None]).sum(1).float()
            ids = torch.arange(kb, end, device=q.device).expand(q.shape[0], -1)
            scores.masked_fill_(ids >= visible[:, None], -torch.inf)
            scores = torch.cat((best_scores, scores), -1)
            ids = torch.cat((best_ids, ids), -1)
            keep = min(512, scores.shape[1])
            order = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :keep]
            best_scores, best_ids = scores.gather(1, order), ids.gather(1, order)
        valid = best_ids < visible[:, None]
        chronological = torch.where(valid, best_ids, 2147483647).sort(dim=-1).values
        result[qb:qb + q.shape[0], :chronological.shape[1]] = torch.where(
            chronological == 2147483647, -1, chronological).to(torch.int32)
    return result, lengths


class CSAIndexer(nn.Module):
    def __init__(self, config, *, device):
        super().__init__()
        self.wq_b = CheckpointLinear(config.q_lora_rank, 64 * 128, device=device)
        self.weights_proj = CheckpointLinear(config.hidden_size, 64, device=device)
        self.compressor = CSACompressor(config, 128, device=device)

    def append_and_select(self, x, qr, state, positions, frequencies):
        compressed = self.compressor.append(x, state.index_compressor, state.length, frequencies)
        begin, end = state.compressed_length, (state.length + x.shape[0]) // 4
        if compressed.shape[0]:
            payload, scales = mxfp4_pack(hadamard(compressed))
            state.index_payload[begin:end].copy_(payload)
            state.index_scales[begin:end].copy_(scales)
        q = self.wq_b(qr).reshape(-1, 64, 128)
        q = mxfp4_simulate(hadamard(rotary(q, positions, frequencies)))
        weights = self.weights_proj(x) * (128**-.5 * 64**-.5)
        indices, lengths = select_csa_topk(q, weights, state, positions, end)
        return indices, lengths, q, weights


class DSV4CSAAttentionBlock(nn.Module):
    """One CSA attention branch, including input norm and output projection.

    forward/prefill: packed BF16 hidden[T,D], states and CPU query_lengths.
    decode: BF16 hidden[B,D], one token per state. forward_hc accepts [T,4,D]
    and adds the official attention-side mHC pre/post mixing. No FFN is run.

    A state belongs to this block and its creating CUDA stream. Reset or fork
    states explicitly; the block never infers request identity from row order.
    """
    def __init__(self, config=None, *, device="cuda", query_chunk_size=128):
        super().__init__()
        self.config = config or DSV4CSAConfig()
        c = self.config
        device = torch.device(device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("DSV4 CSA block requires a CUDA SM120 device")
        if torch.cuda.get_device_capability(device) != (12, 0):
            raise ValueError("DSV4 CSA block requires SM120")
        if type(query_chunk_size) is not int or not 1 <= query_chunk_size <= 1048576:
            raise ValueError("query_chunk_size must be in [1,1048576]")
        self.query_chunk_size = query_chunk_size
        self._state_owner = object()
        self.attn_norm = nn.Parameter(torch.ones(c.hidden_size, device=device, dtype=torch.float32), requires_grad=False)
        self.q_norm = nn.Parameter(torch.ones(c.q_lora_rank, device=device, dtype=torch.float32), requires_grad=False)
        self.kv_norm = nn.Parameter(torch.ones(512, device=device, dtype=torch.float32), requires_grad=False)
        self.attn_sink = nn.Parameter(torch.zeros(64, device=device, dtype=torch.float32), requires_grad=False)
        self.wq_a = CheckpointLinear(c.hidden_size, c.q_lora_rank, device=device)
        self.wq_b = CheckpointLinear(c.q_lora_rank, 64 * 512, device=device)
        self.wkv = CheckpointLinear(c.hidden_size, 512, device=device)
        self.wo_a = CheckpointLinear(64 * 512 // c.o_groups, c.o_groups * c.o_lora_rank, device=device)
        self.wo_b = CheckpointLinear(c.o_groups * c.o_lora_rank, c.hidden_size, device=device)
        self.compressor = CSACompressor(c, 512, device=device)
        self.indexer = CSAIndexer(c, device=device)
        self.register_buffer("frequencies", rope_frequencies(64, c.compress_rope_theta,
            c.original_max_position_embeddings, c.rope_factor, c.beta_fast, c.beta_slow, device), persistent=False)
        mix = (2 + c.hc_mult) * c.hc_mult
        self.hc_attn_fn = nn.Parameter(torch.zeros(mix, c.hc_mult * c.hidden_size, device=device, dtype=torch.float32), requires_grad=False)
        self.hc_attn_base = nn.Parameter(torch.zeros(mix, device=device, dtype=torch.float32), requires_grad=False)
        self.hc_attn_scale = nn.Parameter(torch.ones(3, device=device, dtype=torch.float32), requires_grad=False)
        self.checkpoint_info = {"reference_revision": OFFICIAL_REVISION, "weights": "initialized; not pretrained"}
        self.eval()

    def new_state(self, max_seq_len=4096):
        return DSV4CSAState(self.config, max_seq_len, self.attn_sink.device, self._state_owner)

    def _validate(self, hidden, states, query_lengths, decode):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("stateful CSA block uses host request lengths and cannot be CUDA-graph captured; use the raw fixed-index core for graphs")
        if (hidden.dtype != torch.bfloat16 or hidden.device != self.attn_sink.device or
                hidden.ndim != 2 or hidden.shape[-1] != self.config.hidden_size or not hidden.is_contiguous()):
            raise ValueError("hidden must be contiguous BF16 [T,hidden_size] on the block device")
        if isinstance(query_lengths, torch.Tensor):
            raise ValueError("query_lengths must be CPU Python integers, not a tensor")
        if len(states) != len(query_lengths) or len({id(s) for s in states}) != len(states):
            raise ValueError("provide one distinct state per request")
        if any(type(n) is not int or n < 0 or (decode and n != 1) for n in query_lengths):
            raise ValueError("query lengths must be nonnegative integers; decode requires one per request")
        if sum(query_lengths) != hidden.shape[0]:
            raise ValueError("sum(query_lengths) must equal the number of packed query rows")
        for state, count in zip(states, query_lengths):
            if not isinstance(state, DSV4CSAState) or state._owner is not self._state_owner:
                raise ValueError("state belongs to another attention block")
            state._check_stream()
            if state.device != hidden.device or state._failed:
                raise ValueError("state is on the wrong device or failed; reset before reuse")
            if state.length + count > state.max_seq_len:
                raise ValueError("request exceeds cache capacity")

    @staticmethod
    def _run_core(backend, q, swa, compressed, si, ci, sl, cl, sink, decode):
        kwargs = dict(swa_lengths=sl, compressed_lengths=cl, sink=sink)
        if callable(backend):
            # Explicit diagnostic injection, e.g. an independent server reference.
            return backend(q, swa, compressed, si, ci, **kwargs)
        if backend == "cute":
            from . import sparse_mla_decode, sparse_mla_prefill
            if decode:
                return sparse_mla_decode(q, swa, compressed, si, ci, chunks_per_cta=10, **kwargs)
            return sparse_mla_prefill(q, swa, compressed, si, ci, **kwargs)
        if backend == "flashinfer":
            from flashinfer.mla._sparse_mla_sm120._dsv4_nvfp4 import (
                _nvfp4_sparse_mla_decode, _nvfp4_sparse_mla_prefill)
            args = dict(topk_length=sl, attn_sink=sink, extra_kv_cache=compressed,
                        extra_indices=ci, extra_topk_length=cl)
            if decode:
                return _nvfp4_sparse_mla_decode(q, swa, si, 512**-.5,
                    chunks_per_block_override=10, **args)
            return _nvfp4_sparse_mla_prefill(q, swa, si, 512**-.5, **args)
        raise ValueError("backend must be cute, flashinfer, or an explicit reference callable")

    def _chunk(self, hidden, state, *, backend, decode, trace):
        c, start = self.config, state.length
        count = hidden.shape[0]
        positions = torch.arange(start, start + count, device=hidden.device)
        x = rms_norm(hidden, self.attn_norm, c.rms_norm_eps)
        qr = rms_norm(self.wq_a(x), self.q_norm, c.rms_norm_eps)
        q = self.wq_b(qr).reshape(-1, 64, 512)
        # Follow the official BF16 head-normalization expression. Learned
        # q_norm/kv_norm and compressor pooling independently use FP32.
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + c.rms_norm_eps)
        q = rotary(q, positions, self.frequencies).contiguous()
        kv = rotary(rms_norm(self.wkv(x), self.kv_norm, c.rms_norm_eps), positions, self.frequencies)
        # A temporary pool retains the old ring AND all new query rows. The
        # ring is updated only after attention reads this immutable snapshot.
        new_cache = pack_nvfp4(kv)
        swa = torch.cat((state.swa_cache, new_cache), 0)
        logical = (positions[:, None] - 127).clamp_min(0) + torch.arange(128, device=x.device)
        valid = logical <= positions[:, None]
        physical = torch.where(logical < start, logical % 128, 128 + logical - start)
        si = torch.where(valid, physical, -1).to(torch.int32).contiguous()
        sl = (positions + 1).clamp_max(128).to(torch.int32)

        comp_start = state.compressed_length
        comp = self.compressor.append(x, state.attention_compressor, start, self.frequencies)
        if comp.shape[0]:
            slots = torch.arange(comp_start, comp_start + comp.shape[0], device=x.device, dtype=torch.int32)
            append_nvfp4(comp, state.compressed_cache, slots)
        ci, cl, index_q, index_weights = self.indexer.append_and_select(x, qr, state, positions, self.frequencies)
        core, lse = self._run_core(backend, q, swa, state.compressed_cache, si, ci, sl, cl,
                                   self.attn_sink, decode)
        output = rotary(core, positions, self.frequencies, inverse=True)
        grouped = output.reshape(count, c.o_groups, -1)
        weight = self.wo_a.weight.reshape(c.o_groups, c.o_lora_rank, -1)
        output = torch.einsum("tgd,grd->tgr", grouped, weight)
        output = self.wo_b(output.flatten(1))

        tail = min(count, 128)
        source_slots = torch.arange(count - tail, count, device=x.device)
        destination_slots = (source_slots + start) % 128
        copy_nvfp4_rows(new_cache, source_slots, state.swa_cache, destination_slots)
        state.length += count
        if trace is not None:
            trace.append(dict(start=start, count=count, positions=positions.clone(), query=q.clone(),
                new_kv=kv.clone(), new_compressed=comp.clone(), swa_indices=si.clone(),
                compressed_indices=ci.clone(), swa_lengths=sl.clone(), compressed_lengths=cl.clone(),
                index_query=index_q.clone(), index_weights=index_weights.clone(),
                core_output=core.clone(), lse=lse.clone()))
        return output

    @torch.no_grad()
    def _forward(self, hidden, states, query_lengths, *, backend, decode, trace):
        states, query_lengths = tuple(states), tuple(query_lengths)
        self._validate(hidden, states, query_lengths, decode)
        if not callable(backend) and backend not in ("cute", "flashinfer"):
            raise ValueError("unknown attention backend")
        outputs, offset = [], 0
        for request, (state, length) in enumerate(zip(states, query_lengths)):
            for local in range(0, length, self.query_chunk_size):
                end = min(local + self.query_chunk_size, length)
                try:
                    before = len(trace) if trace is not None else 0
                    outputs.append(self._chunk(hidden[offset + local:offset + end], state,
                        backend=backend, decode=decode, trace=trace))
                    if trace is not None:
                        trace[before]["request"] = request
                except Exception:
                    state._failed = True
                    raise
            offset += length
        return torch.cat(outputs, 0) if outputs else hidden.new_empty(hidden.shape)

    def forward(self, hidden, states: Sequence[DSV4CSAState], query_lengths: Sequence[int],
                *, backend="cute", trace=None):
        return self._forward(hidden, states, query_lengths, backend=backend, decode=False, trace=trace)

    def prefill(self, hidden, states, query_lengths, *, backend="cute", trace=None):
        return self.forward(hidden, states, query_lengths, backend=backend, trace=trace)

    def decode(self, hidden, states, *, backend="cute", trace=None):
        states = tuple(states)
        return self._forward(hidden, states, (1,) * len(states), backend=backend, decode=True, trace=trace)

    @torch.no_grad()
    def forward_hc(self, hidden, states, query_lengths, *, backend="cute", decode=False, trace=None):
        c = self.config
        if hidden.ndim != 3 or hidden.shape[1:] != (c.hc_mult, c.hidden_size):
            raise ValueError("mHC input must be [T,hc_mult,hidden_size]")
        # Validate device/dtype/request metadata before doing any work.
        states, query_lengths = tuple(states), tuple(query_lengths)
        self._validate(hidden[:, 0].contiguous(), states, query_lengths, decode)
        mixed, post, comb = hc_pre(hidden, self.hc_attn_fn, self.hc_attn_scale,
                                  self.hc_attn_base, c.rms_norm_eps, c.hc_eps, c.hc_sinkhorn_iters)
        output = self._forward(mixed.contiguous(), states, query_lengths,
                               backend=backend, decode=decode, trace=trace)
        return hc_post(output, hidden, post, comb)

    @classmethod
    def from_checkpoint(cls, directory, *, layer_id=2, device="cuda", query_chunk_size=128):
        """Load only this attention branch from a local official HF snapshot.

        Supports official layers.N.attn.* keys, FP8 weight/.scale pairs and
        BF16 norm/compressor weights. No network or full-model load occurs.
        """
        from safetensors import safe_open
        directory = Path(directory)
        config = DSV4CSAConfig.from_hf_config(json.loads((directory / "config.json").read_text()), layer_id)
        mapping = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
        result = cls(config, device=device, query_chunk_size=query_chunk_size)
        prefix = f"layers.{layer_id}."

        def tensor(name):
            full = prefix + name
            if full not in mapping:
                raise ValueError(f"missing checkpoint tensor {full}")
            relative = Path(mapping[full])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("checkpoint shard path escapes its directory")
            # HF snapshots normally symlink shards into the sibling blobs
            # directory. Validate the index path, not the symlink target.
            shard = directory / relative
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                return handle.get_tensor(full)

        def copy(parameter, name):
            value = tensor(name)
            if value.shape != parameter.shape:
                raise ValueError(f"wrong checkpoint shape for {name}: {tuple(value.shape)}")
            parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))

        def linear(module, name, *, dequant_bf16=False):
            weight = tensor(name + ".weight")
            scale_name = name + ".scale"
            scale = tensor(scale_name) if prefix + scale_name in mapping else None
            if dequant_bf16 and weight.dtype == torch.float8_e4m3fn:
                if scale is None:
                    raise ValueError(f"missing FP8 scale for {name}")
                weight = dequantize_weight(weight, scale).to(torch.bfloat16)
                scale = None
            module.load_weight(weight, scale)

        for name in ("wq_a", "wq_b", "wkv", "wo_b"):
            linear(getattr(result, name), "attn." + name)
        linear(result.wo_a, "attn.wo_a", dequant_bf16=True)
        copy(result.attn_norm, "attn_norm.weight")
        copy(result.q_norm, "attn.q_norm.weight")
        copy(result.kv_norm, "attn.kv_norm.weight")
        copy(result.attn_sink, "attn.attn_sink")
        linear(result.indexer.wq_b, "attn.indexer.wq_b")
        linear(result.indexer.weights_proj, "attn.indexer.weights_proj")
        for compressor, name in ((result.compressor, "attn.compressor"),
                                 (result.indexer.compressor, "attn.indexer.compressor")):
            linear(compressor.wkv, name + ".wkv")
            linear(compressor.wgate, name + ".wgate")
            copy(compressor.ape, name + ".ape")
            copy(compressor.norm_weight, name + ".norm.weight")
        for name in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale"):
            copy(getattr(result, name), name)
        result.checkpoint_info = dict(reference_revision=OFFICIAL_REVISION,
            weights=str(directory.resolve()), layer_id=layer_id,
            config_sha256=hashlib.sha256((directory / "config.json").read_bytes()).hexdigest())
        return result
