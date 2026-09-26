"""Independent, slow matrix reference for the attention.tex numerical contract.

Server-side test/benchmark support only. Never used by the CUDA operator.
NVFP4 rounding is expressed with an E2M1 codebook, including ties-to-even;
softmax keeps the unquantized FP32 denominator. Disable TF32 when using it.
"""
from __future__ import annotations

import math
import torch


def quantize16(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = x.shape
    grouped = x.float().reshape(*shape[:-1], shape[-1] // 16, 16)
    scales = (grouped.abs().amax(-1) / 6).clamp(max=448).to(torch.float8_e4m3fn)
    sf = scales.float()
    inv = torch.where(sf != 0, sf.reciprocal(), 0)
    normalized = grouped * inv.unsqueeze(-1)
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=x.device)
    boundaries = (levels[1:] + levels[:-1]) / 2
    magnitude = normalized.abs().contiguous()
    code = torch.bucketize(magnitude, boundaries)
    # At a midpoint choose the E2M1 code whose low mantissa bit is even.
    midpoint = boundaries[code.clamp(max=6)]
    code += ((magnitude == midpoint) & (code < 7) & ((code & 1) != 0)).long()
    codes = (code | (torch.signbit(normalized).long() << 3)).to(torch.uint8)
    value = levels[code] * torch.where(torch.signbit(normalized), -1., 1.) * sf.unsqueeze(-1)
    return value.reshape(shape), codes.reshape(shape), scales.view(torch.uint8)


def pack_cache(kv: torch.Tensor) -> torch.Tensor:
    """BF16 [pages,64,512] -> FlashInfer footer-scale uint8 [pages,64,384]."""
    pages = kv.shape[0]
    _, codes, scales = quantize16(kv[..., :448])
    storage = torch.zeros((pages, 64 * 384), dtype=torch.uint8, device=kv.device)
    data = storage[:, :64 * 352].reshape(pages, 64, 352)
    data[..., :224] = codes[..., ::2] | (codes[..., 1::2] << 4)
    data[..., 224:] = kv[..., 448:].to(torch.bfloat16).contiguous().view(torch.uint8)
    storage[:, 64 * 352:].reshape(pages, 64, 32)[..., :28] = scales
    return storage.reshape(pages, 64, 384)


def unpack_cache(cache: torch.Tensor) -> torch.Tensor:
    pages = cache.shape[0]
    storage = cache.reshape(pages, 64 * 384)
    data = storage[:, :64 * 352].reshape(pages, 64, 352)
    codes = torch.stack((data[..., :224] & 15, data[..., :224] >> 4), -1).flatten(-2)
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                          -0., -.5, -1., -1.5, -2., -3., -4., -6.], device=cache.device)
    scales = storage[:, 64 * 352:].reshape(pages, 64, 32)[..., :28].contiguous()
    scales = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
    nope = levels[codes.long()] * scales
    rope = data[..., 224:].contiguous().view(torch.bfloat16).float()
    return torch.cat((nope, rope), -1).reshape(pages * 64, 512)


def plan(batch: int, swa: int, compressed: int, requested: int) -> tuple[int, int]:
    count = max(1, math.ceil(swa / 64) + math.ceil(compressed / 64))
    cpb = min(requested, count) if requested else (min(9, count) if batch <= 64 else count)
    return cpb, math.ceil(count / cpb)


def selected_chunks(cache, indices, lengths):
    indices = indices.reshape(indices.shape[0], -1)
    batch, capacity = indices.shape
    pages = cache.shape[0]
    storage = cache.reshape(pages, 64 * 384)
    for start in range(0, capacity, 64):
        ids = torch.full((batch, 64), -1, dtype=torch.long, device=indices.device)
        take = min(64, capacity - start)
        ids[:, :take] = indices[:, start:start + take]
        position = torch.arange(start, start + 64, device=indices.device)[None, :]
        valid = (ids >= 0) & (ids < pages * 64) & (position < capacity)
        if lengths is not None:
            valid &= position < lengths.clamp(0, capacity)[:, None]
        if pages:
            safe = ids.clamp(0, pages * 64 - 1)
            page, row = safe // 64, safe % 64
            data = storage[page[..., None], row[..., None] * 352 +
                           torch.arange(352, device=cache.device)]
            sf = storage[page[..., None], 64 * 352 + row[..., None] * 32 +
                         torch.arange(28, device=cache.device)].contiguous()
            codes = torch.stack((data[..., :224] & 15, data[..., :224] >> 4), -1).flatten(-2)
            levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                                   -0., -.5, -1., -1.5, -2., -3., -4., -6.], device=cache.device)
            nope = levels[codes.long()] * sf.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
            rope = data[..., 224:].contiguous().view(torch.bfloat16).float()
            kv = torch.cat((nope, rope), -1)
            kv = torch.where(valid[..., None], kv, 0)
        else:
            kv = torch.zeros((batch, 64, 512), device=indices.device)
        yield kv, valid


@torch.no_grad()
def reference(query, swa_cache, compressed_cache, swa_indices, compressed_indices,
              *, swa_lengths=None, compressed_lengths=None, sink=None,
              chunks_per_cta=0, softmax_scale=512**-.5, lse_scale=1.):
    original_shape = query.shape
    q = query.reshape(-1, 64, 512).float()
    batch = q.shape[0]
    qn, _, _ = quantize16(q[..., :448])
    chunks = list(selected_chunks(swa_cache, swa_indices, swa_lengths))
    chunks += list(selected_chunks(compressed_cache, compressed_indices, compressed_lengths))
    cpb, splits = plan(batch, swa_indices.shape[-1], compressed_indices.shape[-1], chunks_per_cta)
    partials, lses, raw_outputs, raw_denominators = [], [], [], []
    for split in range(splits):
        m = torch.full((batch, 64), -1.e30, device=q.device)
        ell = torch.zeros_like(m)
        accum = torch.zeros_like(q)
        for kv, valid in chunks[split * cpb:(split + 1) * cpb]:
            z = (qn @ kv[..., :448].transpose(-1, -2) +
                 q[..., 448:] @ kv[..., 448:].transpose(-1, -2)) * (softmax_scale * math.log2(math.e))
            z = z.masked_fill(~valid[:, None, :], -1.e30)
            local_z = z.reshape(batch, 64, 8, 8)
            local_max = local_z.amax(-1)
            local_p = torch.exp2(local_z - local_max[..., None])
            local_p *= valid.reshape(batch, 1, 8, 8)
            block_max = local_max.amax(-1)
            block_sum = (local_p.sum(-1) * torch.exp2(local_max - block_max[..., None])).sum(-1)
            next_m = torch.maximum(m, block_max)
            alpha = torch.where(m > -1.e29, torch.exp2(m - next_m), 0)
            ell = alpha * ell + block_sum * torch.exp2(block_max - next_m)
            w = (local_p * torch.exp2(local_max - next_m[..., None])[..., None])
            w = w.reshape(batch, 64, 64).to(torch.bfloat16).float()
            probability, _, _ = quantize16(w)
            vt, _, _ = quantize16(kv[..., :448].transpose(-1, -2).contiguous())
            contribution = torch.cat((probability @ vt.transpose(-1, -2), w @ kv[..., 448:]), -1)
            accum = alpha[..., None] * accum + contribution
            m = next_m
        lam = torch.where(ell > 0, m + ell.log2(), -1.e30)
        normalized = accum / torch.where(ell > 0, ell, 1)[..., None]
        partials.append(normalized.to(torch.bfloat16).float())
        raw_outputs.append(accum)
        raw_denominators.append(ell)
        lses.append(lam)
    lam = torch.stack(lses, -1)
    beta = sink.float()[None, :] * math.log2(math.e) if sink is not None else torch.full_like(lses[0], -torch.inf)
    maximum = torch.maximum(lam.amax(-1), beta)
    mass = torch.where(lam > -1.e29, torch.exp2(lam - maximum[..., None]), 0)
    total = mass.sum(-1) + torch.exp2(beta - maximum)
    safe_total = torch.where(total > 0, total, 1)
    if splits == 1:
        ell = raw_denominators[0]
        inv = torch.where(ell > 0, ell.reciprocal(), 0)
        factor = inv * (mass[..., 0] / safe_total)
        out = raw_outputs[0] * factor[..., None]
    else:
        out = (torch.stack(partials, -2) * mass[..., None]).sum(-2) / safe_total[..., None]
    final_lse = torch.where(total > 0, (maximum + total.log2()) * lse_scale, -torch.inf)
    return out.to(torch.bfloat16).reshape(original_shape), final_lse


def make_problem(batch=2, swa=128, compressed=512, seed=43, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(seed)
    def vectors(shape):
        x = torch.randn(shape, generator=generator, device=device) * .25
        # Vary scales both between tokens and between channel groups.
        channels = torch.linspace(.25, 2., 32, device=device).repeat_interleave(16)
        tokens = torch.linspace(.4, 2., shape[-2], device=device)[..., None]
        return (x * channels * tokens).to(torch.bfloat16)
    query = vectors((batch, 64, 512))
    swa_pages = max(1, math.ceil(max(1, swa) / 64)) * batch
    comp_pages = max(2, math.ceil(max(1, compressed) / 64) + 2) * batch
    cache1 = pack_cache(vectors((swa_pages, 64, 512)))
    cache2 = pack_cache(vectors((comp_pages, 64, 512)))
    ids1 = torch.randint(swa_pages * 64, (batch, swa), generator=generator, device=device, dtype=torch.int32)
    ids2 = torch.randint(comp_pages * 64, (batch, compressed), generator=generator, device=device, dtype=torch.int32)
    return query, cache1, cache2, ids1, ids2


def prefill_reference(query, swa_cache, compressed_cache, swa_indices,
                      compressed_indices, **kwargs):
    """The same quantized matrix reference with one complete stream/query."""
    chunks = max(1, math.ceil(swa_indices.shape[-1] / 64) +
                 math.ceil(compressed_indices.shape[-1] / 64))
    return reference(query, swa_cache, compressed_cache, swa_indices,
                     compressed_indices, chunks_per_cta=chunks, **kwargs)


def make_prefill_problem(queries=7, swa=128, compressed=512, seed=91, device="cuda"):
    """Independent query rows sharing two pools; selection is synthetic.

    This is a fixed-index core fixture, not a CSA compressor/indexer model.
    Cache allocation is independent of the number of query rows.
    """
    _, cache1, cache2, _, _ = make_problem(1, swa, compressed, seed, device)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    query = (torch.randn((queries, 64, 512), generator=generator, device=device) * .25)
    query *= torch.linspace(.25, 2., 32, device=device).repeat_interleave(16)
    ids1 = torch.randint(cache1.shape[0] * 64, (queries, swa),
                         generator=generator, device=device, dtype=torch.int32)
    ids2 = torch.randint(cache2.shape[0] * 64, (queries, compressed),
                         generator=generator, device=device, dtype=torch.int32)
    return query.to(torch.bfloat16), cache1, cache2, ids1, ids2
