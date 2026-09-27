#!/usr/bin/env python3
"""Single-GPU expert compute and routing overhead; no distributed transport.

Uses per-expert constant weights to permit an independent analytic reference.
This is an operator fixture, not DS-V4 checkpoint/quality validation.
"""
from __future__ import annotations

import argparse

import torch
import sm120_nvfp4

from operator_benchmark_utils import (add_timing_arguments, environment, errors,
                                      measure, run_cases, validate_timing)


def quantized_reference(x):
    """MoE quantization preserves tiny nonzero scales; attention does not."""
    groups = x.float().reshape(x.shape[0], -1, 16)
    maximum = groups.abs().amax(-1)
    scale = (maximum / 6).clamp_max(448).to(torch.float8_e4m3fn).float()
    scale = torch.where((maximum > 0) & (scale == 0), 2.**-9, scale)
    normalized = groups / torch.where(scale > 0, scale, 1)[..., None]
    # Even encodings precede odd encodings, implementing ties-to-even.
    levels = torch.tensor([0., 1., 2., 4., .5, 1.5, 3., 6.], device=x.device)
    nearest = (normalized.abs()[..., None] - levels).abs().argmin(-1)
    signs = torch.where(torch.signbit(normalized), -1., 1.)
    return (levels[nearest] * signs * scale[..., None]).reshape_as(x)


@torch.no_grad()
def case(spec, args):
    rows, experts = spec["rows"], args.experts
    hidden, intermediate = args.hidden, args.intermediate
    if spec["routing"] == "balanced":
        counts = [rows // experts + (i < rows % experts) for i in range(experts)]
    else:
        hot = max(1, rows * 4 // 5)
        counts = [hot] + [(rows - hot) // (experts - 1) + (i < (rows - hot) % (experts - 1))
                          for i in range(experts - 1)]
    prefix = [0]
    for count in counts:
        prefix.append(prefix[-1] + count)
    lengths = torch.tensor(counts, device="cuda", dtype=torch.int32)
    offsets = torch.tensor(prefix, device="cuda", dtype=torch.int32)
    owner = torch.repeat_interleave(torch.arange(experts, device="cuda"), lengths.long()).int()
    pad = ((max(counts) + 127) // 128) * 128
    global_pad = ((rows + 127) // 128) * 128
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    x = (torch.randn((rows, hidden), generator=generator, device="cuda") * .25).bfloat16()
    packed = torch.empty((rows, hidden // 2), device="cuda", dtype=torch.uint8)
    scales = torch.zeros((experts, pad * sm120_nvfp4.scale_k_padded(hidden)), device="cuda", dtype=torch.uint8)
    gate = torch.full((experts, 2 * intermediate, hidden // 2), 0x11, device="cuda", dtype=torch.uint8)
    down = torch.full((experts, hidden, intermediate // 2), 0x11, device="cuda", dtype=torch.uint8)
    gs = torch.empty((experts, sm120_nvfp4.scale_b_elements(1, 2 * intermediate, hidden)), device="cuda", dtype=torch.uint8)
    ds = torch.full((experts, sm120_nvfp4.scale_b_elements(1, hidden, intermediate)), 0x10, device="cuda", dtype=torch.uint8)
    for expert in range(experts):
        gs[expert].fill_(0x10 + (expert % 3) * 4)
    output = torch.empty((rows, hidden), device="cuda", dtype=torch.float16)
    workspace = torch.empty(sm120_nvfp4.expert_moe_workspace_bytes(rows, hidden,
        intermediate, experts, pad), device="cuda", dtype=torch.uint8)

    def quantize():
        return sm120_nvfp4.quantize_expert(x, lengths, offsets, scale_m_pad=pad,
                                          output=packed, output_scale=scales)

    quantize()

    def expert_compute():
        return (sm120_nvfp4.expert_moe(packed, scales, gate, gs, down, ds, lengths,
                                     offsets, output=output, workspace=workspace),)

    def quantize_and_compute():
        quantize()
        return expert_compute()

    expert_compute()
    # For constant weights all output channels share the same dot product.
    # Reproduce FP16 gate/up, FP32 SiLU, MoE NVFP4 rounding, and FP16 down output.
    dequantized = quantized_reference(x)
    weight_scale = gs[:, :1].contiguous().view(torch.float8_e4m3fn).float().flatten()
    gate_value = (dequantized.sum(-1) * (.5 * weight_scale[owner.long()])).half().float()
    activation = gate_value / (1 + torch.exp(-gate_value)) * gate_value
    act_q = quantized_reference(activation[:, None].expand(-1, 16).contiguous())[:, 0]
    expected = (act_q * (intermediate * .5 * 2.**-5)).half()[:, None].expand_as(output)
    torch.testing.assert_close(output, expected, rtol=.02, atol=.005)
    accuracy = errors(output, expected)
    if accuracy["rmse"] > .003 or not torch.isfinite(output).all().item():
        raise AssertionError("analytic MoE reference RMSE exceeds .003 or output is nonfinite")

    # Same expanded expert-major rows, deliberately repeating local routing.
    # This measures the saved preparation work, not a competing GEMM backend.
    one_len = torch.tensor([rows], device="cuda", dtype=torch.int32)
    one_offset = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    global_packed = torch.empty_like(packed)
    global_scales = torch.zeros((1, global_pad * sm120_nvfp4.scale_k_padded(hidden)), device="cuda", dtype=torch.uint8)
    route_weights = torch.ones((rows, 1), device="cuda")
    route_ids = owner[:, None].contiguous()
    routed_output = torch.empty_like(output)

    def reroute():
        sm120_nvfp4.quantize_expert(x, one_len, one_offset, scale_m_pad=global_pad,
                                  output=global_packed, output_scale=global_scales)
        return (sm120_nvfp4.fused_moe(global_packed, global_scales.flatten(), gate, gs,
            down, ds, route_ids, route_weights, output=routed_output),)

    reroute()
    torch.testing.assert_close(output, routed_output, rtol=0, atol=0)
    result = dict(**spec, hidden=hidden, intermediate=intermediate, experts=experts,
        expert_rows=counts, accuracy=accuracy, routing_max_abs=0.,
        input_kind="synthetic BF16 expert-major rows, constant E2M1=.5 weights, per-expert scales",
        scope="single GPU expanded rows; no dispatch/combine communication or model router",
        workspace_bytes=workspace.numel(),
        weights_bytes=gate.numel() + down.numel() + gs.numel() + ds.numel(),
        activation_quantize=measure(quantize, args),
        expert_compute_prequantized=measure(expert_compute, args),
        quantize_and_expert_compute=measure(quantize_and_compute, args),
        quantize_and_repeated_routing=measure(reroute, args))
    result["handoff_speedup"] = result["quantize_and_repeated_routing"]["p50_us"] / result["quantize_and_expert_compute"]["p50_us"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[16, 128, 512])
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--routing", nargs="+", choices=("balanced", "skewed"), default=["balanced", "skewed"])
    add_timing_arguments(parser)
    args = parser.parse_args()
    validate_timing(parser, args)
    if (min(args.rows) < 1 or max(args.rows) > 1048576 or not 2 <= args.experts <= 256 or
            min(args.hidden, args.intermediate) < 32 or args.hidden % 32 or args.intermediate % 32):
        parser.error("invalid rows/experts/aligned dimensions")
    if args.flashinfer:
        parser.error("this benchmark compares handoff/control paths; FlashInfer MoE is not a matched baseline here")
    cases = [dict(rows=rows, routing=routing) for rows in args.rows for routing in args.routing]
    run_cases(args, environment(__file__), cases, lambda spec: case(spec, args))


if __name__ == "__main__":
    main()
