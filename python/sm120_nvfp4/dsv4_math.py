"""GPU tensor primitives for the DS-V4 CSA block (no attention fallback).

Model semantics: deepseek-ai/DeepSeek-V4-Flash, revision
60d8d70770c6776ff598c94bb586a859a38244f1, inference/model.py and kernel.py.
Main attention/cache use this repository's NVFP4 contract. The independent
indexer uses MXFP4: groups of 32, E2M1 payload and power-of-two scales.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


OFFICIAL_REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"


def rms_norm(x, weight, eps):
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
    return (value * weight.float()).to(x.dtype)


def rope_frequencies(dim, base, original_length, factor, beta_fast, beta_slow, device):
    indices = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    frequencies = 1. / (base ** (indices / dim))
    if original_length:
        correction = lambda rotations: dim * math.log(original_length / (rotations * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(correction(beta_fast)), 0)
        high = min(math.ceil(correction(beta_slow)), dim - 1)
        width = high - low if high != low else .001
        smooth = 1 - ((torch.arange(dim // 2, device=device) - low) / width).clamp(0, 1)
        frequencies = frequencies / factor * (1 - smooth) + frequencies * smooth
    return frequencies


def rotary(x, positions, frequencies, inverse=False):
    """Adjacent real/imag pairs, last 64 channels; preserve BF16 rounding."""
    rd = frequencies.numel() * 2
    angles = positions.float()[:, None] * frequencies[None]
    shape = (positions.numel(),) + (1,) * (x.ndim - 2) + (rd // 2,)
    cos, sin = angles.cos().reshape(shape), angles.sin().reshape(shape)
    if inverse:
        sin = -sin
    pair = x[..., -rd:].float().reshape(*x.shape[:-1], rd // 2, 2)
    result = x.clone()
    rotated = torch.stack((pair[..., 0] * cos - pair[..., 1] * sin,
                           pair[..., 1] * cos + pair[..., 0] * sin), -1)
    result[..., -rd:] = rotated.flatten(-2).to(x.dtype)
    return result


def hadamard(x):
    """Normalized Sylvester transform, matching the official 128-D rotation."""
    dim = x.shape[-1]
    if dim <= 0 or dim & (dim - 1):
        raise ValueError("Hadamard dimension must be a power of two")
    y = x.float()
    width = 1
    while width < dim:
        pairs = y.reshape(*x.shape[:-1], -1, 2, width)
        a, b = pairs[..., 0, :], pairs[..., 1, :]
        y = torch.stack((a + b, a - b), -2).reshape_as(y)
        width *= 2
    return (y * dim**-.5).to(x.dtype)


def mxfp4_pack(x):
    """Official indexer quantization, not main attention's E4M3/16 format."""
    groups = x.float().reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    amax = groups.abs().amax(-1).clamp_min(6 * 2.**-126)
    exponent = torch.ceil(torch.log2(amax / 6)).clamp(-126, 127)
    scale = torch.exp2(exponent)
    normalized = (groups / scale[..., None]).clamp(-6, 6)
    levels = x.new_tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float32)
    midpoint = (levels[:-1] + levels[1:]) * .5
    magnitude = normalized.abs().contiguous()
    codes = torch.bucketize(magnitude, midpoint)
    ties = (magnitude == midpoint[codes.clamp_max(6)]) & (codes < 7) & ((codes & 1) != 0)
    codes = (codes + ties.long()) | (torch.signbit(normalized).long() << 3)
    codes = codes.to(torch.uint8).reshape_as(x)
    payload = codes[..., ::2] | (codes[..., 1::2] << 4)
    return payload.contiguous(), (exponent + 127).to(torch.uint8).contiguous()


def mxfp4_unpack(payload, scales):
    codes = torch.stack((payload & 15, payload >> 4), -1).flatten(-2).long()
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                           -0., -.5, -1., -1.5, -2., -3., -4., -6.], device=payload.device)
    sf = torch.exp2(scales.float() - 127).repeat_interleave(32, -1)
    return (levels[codes] * sf).to(torch.bfloat16)


def mxfp4_simulate(x):
    return mxfp4_unpack(*mxfp4_pack(x))


def append_nvfp4(values, cache, slots):
    return torch.ops.sm120_nvfp4.sparse_mla_pack_cache(values.contiguous(), slots.contiguous(), cache)


def pack_nvfp4(values):
    rows = values.shape[0]
    cache = torch.zeros(((rows + 63) // 64, 64, 384), dtype=torch.uint8, device=values.device)
    slots = torch.arange(rows, dtype=torch.int32, device=values.device)
    return append_nvfp4(values, cache, slots)


def copy_nvfp4_rows(source, source_slots, destination, destination_slots):
    """Copy packed rows without a second quantization. Destinations are unique."""
    src = source.reshape(-1, 64 * 384)
    dst = destination.reshape(-1, 64 * 384)
    sp, sr = source_slots.long() // 64, source_slots.long() % 64
    dp, dr = destination_slots.long() // 64, destination_slots.long() % 64
    data = torch.arange(352, device=source.device)
    scale = torch.arange(32, device=source.device)
    dst[dp[:, None], dr[:, None] * 352 + data] = src[sp[:, None], sr[:, None] * 352 + data]
    dst[dp[:, None], 64 * 352 + dr[:, None] * 32 + scale] = src[sp[:, None], 64 * 352 + sr[:, None] * 32 + scale]


def dequantize_weight(weight, scale):
    expected = ((weight.shape[0] + 127) // 128, (weight.shape[1] + 127) // 128)
    if weight.dtype != torch.float8_e4m3fn or tuple(scale.shape) != expected:
        raise ValueError("FP8 weights require E4M3 [out,in] and per-128x128 scales")
    expanded = scale.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
    return weight.float() * expanded[:weight.shape[0], :weight.shape[1]]


class CheckpointLinear(nn.Module):
    """BF16/FP32 projection, or the official FP8 activation/weight contract.

    FP8 weights stay packed. The functional FP8 path uses PyTorch FP32 GEMMs
    per K=128 group and applies scales after each partial dot product. It is
    intended for block integration/validation, not a performance claim.
    """
    def __init__(self, in_features, out_features, *, device, dtype=torch.bfloat16):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.register_buffer("scale", None)

    def load_weight(self, weight, scale=None):
        if tuple(weight.shape) != (self.out_features, self.in_features):
            raise ValueError(f"projection weight has shape {tuple(weight.shape)}, expected {(self.out_features, self.in_features)}")
        if weight.dtype == torch.float8_e4m3fn:
            expected = ((self.out_features + 127) // 128, (self.in_features + 127) // 128)
            if scale is None or tuple(scale.shape) != expected or self.in_features % 128:
                raise ValueError("FP8 projection requires per-128x128 scales and aligned input dimension")
            if not scale.is_floating_point():
                raise ValueError("FP8 scales must contain numeric floating-point values, not encoded uint8 bytes")
            self.scale = scale.to(device=self.weight.device).contiguous()
            target_dtype = weight.dtype
        else:
            if scale is not None or weight.dtype not in (torch.bfloat16, torch.float32, torch.float16):
                raise ValueError("unquantized projection must be floating point without scales")
            self.scale = None
            target_dtype = self.weight.dtype
        self.weight = nn.Parameter(weight.to(device=self.weight.device, dtype=target_dtype).contiguous(), requires_grad=False)

    def forward(self, x):
        if self.scale is None:
            return F.linear(x.to(self.weight.dtype), self.weight)
        shape = x.shape[:-1]
        grouped = x.float().reshape(-1, self.in_features // 128, 128)
        sf = torch.exp2(torch.ceil(torch.log2(grouped.abs().amax(-1).clamp_min(1.e-4) / 448)))
        encoded = (grouped / sf[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).float()
        output = torch.zeros((grouped.shape[0], self.out_features), device=x.device, dtype=torch.float32)
        for k in range(self.in_features // 128):
            product = F.linear(encoded[:, k], self.weight[:, k * 128:(k + 1) * 128].float())
            ws = self.scale[:, k].float().repeat_interleave(128)[:self.out_features]
            output.add_(product * (sf[:, k, None] * ws[None]))
        return output.to(torch.bfloat16).reshape(*shape, self.out_features)


def hc_pre(x, fn, scale, base, eps, sinkhorn_eps, iterations):
    """Official attention-side mHC pre-mix; returns mixed x, post and comb."""
    count = x.shape[-2]
    flat = x.flatten(-2).float()
    mixes = F.linear(flat, fn.float()) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)
    pre = torch.sigmoid(mixes[..., :count] * scale[0] + base[:count]) + sinkhorn_eps
    post = 2 * torch.sigmoid(mixes[..., count:2 * count] * scale[1] + base[count:2 * count])
    comb = (mixes[..., 2 * count:] * scale[2] + base[2 * count:]).reshape(-1, count, count)
    comb = torch.softmax(comb, -1) + sinkhorn_eps
    comb = comb / (comb.sum(-2, keepdim=True) + sinkhorn_eps)
    for _ in range(iterations - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + sinkhorn_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + sinkhorn_eps)
    mixed = (pre[..., None] * x.float()).sum(-2).to(x.dtype)
    return mixed, post, comb


def hc_post(output, residual, post, comb):
    return (post[..., None] * output.float().unsqueeze(-2) +
            (comb[..., None] * residual.float().unsqueeze(-2)).sum(-3)).to(output.dtype)
