"""Stateless, full-prefix model reference used only by GPU server tests.

Rebuilds compression windows and candidate sets from hidden states, without
reading production compressor state or production selected indices. Attention
uses the separately established attention.tex matrix reference.
"""
import math

import torch
import torch.nn.functional as F

from sparse_mla_reference import pack_cache, prefill_reference


def norm_ref(x, weight, eps):
    value = x.float()
    return (value / torch.sqrt(value.square().mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


def rope_ref(x, positions, frequencies, inverse=False):
    rd = frequencies.numel() * 2
    z = torch.view_as_complex(x[..., -rd:].float().reshape(*x.shape[:-1], rd // 2, 2))
    phase = torch.polar(torch.ones((len(positions), rd // 2), device=x.device),
                        positions.float()[:, None] * frequencies[None])
    if inverse:
        phase = phase.conj()
    if x.ndim == 3:
        phase = phase[:, None]
    return torch.cat((x[..., :-rd], torch.view_as_real(z * phase).flatten(-2).to(x.dtype)), -1)


def mxfp4_ref(x):
    groups = x.float().reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    value = groups.abs().amax(-1).clamp_min(6 * 2.**-126) / 6
    mantissa, exponent = torch.frexp(value)
    scale = torch.ldexp(torch.ones_like(value), exponent - (mantissa == .5).int())
    normalized = groups / scale[..., None]
    # Nearest codebook entry; even encodings precede odd encodings on ties.
    levels = torch.tensor([0., 1., 2., 4., .5, 1.5, 3., 6.], device=x.device)
    closest = (normalized.abs()[..., None] - levels).abs().argmin(-1)
    signs = torch.where(torch.signbit(normalized), -1., 1.)
    return (levels[closest] * signs * scale[..., None]).reshape_as(x).to(x.dtype)


def hadamard_ref(x):
    h = torch.ones((1, 1), device=x.device)
    while h.shape[0] < x.shape[-1]:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return (x.float() @ h / math.sqrt(x.shape[-1])).to(x.dtype)


def pack_rows_ref(x):
    padded = F.pad(x, (0, 0, 0, (-x.shape[0]) % 64))
    return pack_cache(padded.reshape(-1, 64, 512))


def linear_ref(x, module):
    if module.scale is None:
        return F.linear(x.to(module.weight.dtype), module.weight)
    grouped = x.float().reshape(-1, module.in_features // 128, 128)
    scale = 2. ** torch.ceil(torch.log2(grouped.abs().amax(-1).clamp_min(1.e-4) / 448))
    quantized = (grouped / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    activation = (quantized * scale[..., None]).reshape(-1, module.in_features)
    ws = module.scale.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
    weight = module.weight.float() * ws[:module.out_features, :module.in_features]
    return F.linear(activation, weight).to(torch.bfloat16).reshape(*x.shape[:-1], module.out_features)


def compressor_ref(x, compressor, frequencies):
    dim = compressor.dim
    values = F.linear(x.float(), compressor.wkv.weight.float())
    scores = F.linear(x.float(), compressor.wgate.weight.float())
    scores += compressor.ape[torch.arange(x.shape[0], device=x.device) % 4]
    outputs = []
    for end in range(4, x.shape[0] + 1, 4):
        current_v, current_s = values[end - 4:end, dim:], scores[end - 4:end, dim:]
        if end > 4:
            vv = torch.cat((values[end - 8:end - 4, :dim], current_v), 0)
            ss = torch.cat((scores[end - 8:end - 4, :dim], current_s), 0)
        else:
            vv, ss = current_v, current_s  # missing prefix has exactly zero mass
        pooled = (ss.softmax(0) * vv).sum(0).to(torch.bfloat16)[None]
        pooled = norm_ref(pooled, compressor.norm_weight, compressor.eps)
        outputs.append(rope_ref(pooled, torch.tensor([end - 4], device=x.device), frequencies))
    output = torch.cat(outputs) if outputs else x.new_empty((0, dim))
    return output, values, scores


@torch.no_grad()
def full_prefix_reference(block, hidden, *, core_reference=prefill_reference):
    c = block.config
    positions = torch.arange(hidden.shape[0], device=hidden.device)
    x = norm_ref(hidden, block.attn_norm, c.rms_norm_eps)
    qr = norm_ref(linear_ref(x, block.wq_a), block.q_norm, c.rms_norm_eps)
    q = linear_ref(qr, block.wq_b).reshape(-1, 64, 512)
    q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + c.rms_norm_eps)
    q = rope_ref(q, positions, block.frequencies).contiguous()
    kv = rope_ref(norm_ref(linear_ref(x, block.wkv), block.kv_norm, c.rms_norm_eps), positions, block.frequencies)
    compressed, cv, cs = compressor_ref(x, block.compressor, block.frequencies)
    index_k, iv, iscores = compressor_ref(x, block.indexer.compressor, block.frequencies)
    index_k = mxfp4_ref(hadamard_ref(index_k))
    index_q = linear_ref(qr, block.indexer.wq_b).reshape(-1, 64, 128)
    index_q = mxfp4_ref(hadamard_ref(rope_ref(index_q, positions, block.frequencies)))
    weights = linear_ref(x, block.indexer.weights_proj) * (128**-.5 * 64**-.5)
    available = len(index_k)
    ids = torch.full((len(x), 512), -1, device=x.device, dtype=torch.int32)
    if available:
        scores = (torch.einsum("qhd,kd->qhk", index_q, index_k).relu() * weights[..., None]).sum(1).float()
        logical = torch.arange(available, device=x.device)[None]
        scores.masked_fill_(logical >= ((positions + 1) // 4)[:, None], -torch.inf)
        chosen = scores.argsort(dim=-1, descending=True, stable=True)[:, :min(512, available)]
        chosen = torch.where(chosen < ((positions + 1) // 4)[:, None], chosen, 2147483647)
        chosen = chosen.sort(-1).values
        ids[:, :chosen.shape[-1]] = torch.where(chosen == 2147483647, -1, chosen).int()
    swa_ids = (positions[:, None] - 127).clamp_min(0) + torch.arange(128, device=x.device)[None]
    swa_ids = torch.where(swa_ids <= positions[:, None], swa_ids, -1).int()
    sl, cl = (positions + 1).clamp_max(128).int(), ((positions + 1) // 4).clamp_max(512).int()
    core, lse = core_reference(q, pack_rows_ref(kv), pack_rows_ref(compressed),
                               swa_ids, ids, swa_lengths=sl, compressed_lengths=cl, sink=block.attn_sink)
    out = rope_ref(core, positions, block.frequencies, inverse=True).reshape(len(x), c.o_groups, -1)
    out = torch.einsum("tgd,grd->tgr", out, block.wo_a.weight.reshape(c.o_groups, c.o_lora_rank, -1))
    out = linear_ref(out.flatten(1), block.wo_b)
    return out, dict(query=q, kv=kv, compressed=compressed, index_keys=index_k,
        index_query=index_q, indices=ids, swa_lengths=sl, compressed_lengths=cl,
        compressor_values=cv, compressor_scores=cs, index_values=iv, index_scores=iscores,
        core_output=core, lse=lse)
