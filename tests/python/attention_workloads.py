"""Synthetic CSA/HCA core fixtures, with causal positions and disjoint requests.

These are post-compression/post-RoPE vectors, not model compressor outputs.
Only fixture preparation uses host loops; measured operators consume GPU indices.
"""
import math
import random

import torch

from sparse_mla_reference import pack_cache


def attention_workload(batch, queries, history, kind="csa", *, seed=43,
                       mixed_lengths=False, selection="random", device="cuda"):
    if min(batch, queries) < 1 or history < 0 or kind not in ("csa", "hca"):
        raise ValueError("invalid attention workload")
    if selection not in ("random", "contiguous"):
        raise ValueError("selection must be random or contiguous")
    ratio = 4 if kind == "csa" else 128
    histories = ([history * i // max(1, batch - 1) for i in range(batch)]
                 if mixed_lengths and batch > 1 else [history] * batch)
    swa_stride = math.ceil((min(history, 127) + queries) / 64) * 64
    comp_stride = max(64, math.ceil(((history + queries) // ratio) / 64) * 64)
    capacity = 512 if kind == "csa" else (history + queries) // ratio
    if batch * max(swa_stride, comp_stride) >= 2**31:
        raise ValueError("physical slots exceed int32")
    generator = torch.Generator(device=device).manual_seed(seed)

    def vectors(shape):
        x = torch.randn(shape, generator=generator, device=device)
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1.e-6)).to(torch.bfloat16)

    def cache(stride):
        out = torch.empty((batch * stride // 64, 64, 384), device=device, dtype=torch.uint8)
        for b in range(batch):
            first, last = b * stride // 64, (b + 1) * stride // 64
            out[first:last] = pack_cache(vectors((stride // 64, 64, 512)))
        return out

    q = vectors((batch * queries, 64, 512))
    swa, comp = cache(swa_stride), cache(comp_stride)
    si, ci, sl, cl, positions = [], [], [], [], []
    rng = random.Random(seed)
    for b, prefix in enumerate(histories):
        base = max(0, prefix - 127)
        for position in range(prefix, prefix + queries):
            window = range(max(0, position - 127), position + 1)
            sw = [b * swa_stride + p - base for p in window]
            visible = (position + 1) // ratio
            count = min(512, visible) if kind == "csa" else visible
            if kind == "csa" and selection == "random":
                chosen = sorted(rng.sample(range(visible), count))
            else:
                chosen = list(range(visible - count, visible))
            si.append(sw + [-1] * (128 - len(sw)))
            ci.append([b * comp_stride + p for p in chosen] + [-1] * (capacity - count))
            sl.append(len(sw))
            cl.append(count)
            positions.append(position)
    integer = lambda data: torch.tensor(data, device=device, dtype=torch.int32)
    sink = torch.randn(64, generator=generator, device=device)
    inputs = (q, swa, comp, integer(si), integer(ci).reshape(batch * queries, capacity))
    kwargs = dict(swa_lengths=integer(sl), compressed_lengths=integer(cl), sink=sink)
    metadata = dict(kind=kind, requests=batch, queries_per_request=queries,
        history_tokens=histories, compression_ratio=ratio, query_positions=positions,
        swa_rows_per_query=sl, selected_compressed_rows_per_query=cl,
        swa_slots_per_request=swa_stride, compressed_slots_per_request=comp_stride,
        compressed_candidate_capacity=capacity,
        selection="all visible entries" if kind == "hca" else selection + ", unique, chronological",
        input_kind="synthetic unit-RMS post-compression/post-RoPE; random sink")
    return inputs, kwargs, metadata
