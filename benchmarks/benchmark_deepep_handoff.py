#!/usr/bin/env python3
"""Two-GPU DeepEP-compatible dispatch -> NVFP4 expert -> combine benchmark.

This benchmark uses a transparent torch.distributed reference dispatcher, not
DeepEP's native transport kernels. It measures the integration contract:
expert-major expanded layout, route-handle reuse, BF16-to-NVFP4 quantization,
persistent grouped expert compute, and reverse combine.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist

import sm120_nvfp4


@dataclass
class RouteHandle:
    token_order: torch.Tensor
    flat_order: torch.Tensor
    send_splits: list[int]
    recv_splits: list[int]
    expert_order: torch.Tensor
    grouped_metadata: torch.Tensor
    seqlens: torch.Tensor
    cu_seqlens: torch.Tensor
    seqlens_list: list[int]
    return_order: torch.Tensor
    return_send_splits: list[int]
    return_recv_splits: list[int]
    returned_metadata: torch.Tensor

    def bytes(self) -> int:
        tensors = (
            self.token_order,
            self.flat_order,
            self.expert_order,
            self.grouped_metadata,
            self.seqlens,
            self.cu_seqlens,
            self.return_order,
            self.returned_metadata,
        )
        return sum(t.numel() * t.element_size() for t in tensors)


def exchange_counts(send_counts: torch.Tensor) -> tuple[list[int], list[int]]:
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts)
    return send_counts.cpu().tolist(), recv_counts.cpu().tolist()


def exchange_tensor(
    send: torch.Tensor,
    send_splits: list[int],
    recv_splits: list[int],
    recv: torch.Tensor | None = None,
) -> torch.Tensor:
    if recv is None:
        recv = torch.empty(
            (sum(recv_splits),) + tuple(send.shape[1:]),
            dtype=send.dtype,
            device=send.device,
        )
    dist.all_to_all_single(
        recv,
        send.contiguous(),
        output_split_sizes=recv_splits,
        input_split_sizes=send_splits,
    )
    return recv


def make_topk(
    tokens: int,
    topk: int,
    num_experts: int,
    rank: int,
    routing: str,
    device: torch.device,
) -> torch.Tensor:
    token_ids = torch.arange(tokens, device=device, dtype=torch.int64)
    slot_ids = torch.arange(topk, device=device, dtype=torch.int64)
    if routing == "balanced":
        return (token_ids[:, None] * topk + slot_ids[None, :] + rank) % num_experts

    cutoff = int(tokens * 0.8)
    experts_per_rank = num_experts // 2
    hot = slot_ids % experts_per_rank
    cold = experts_per_rank + slot_ids % experts_per_rank
    return torch.where(
        (token_ids < cutoff).unsqueeze(1),
        hot.unsqueeze(0),
        cold.unsqueeze(0),
    )


def build_route_handle(
    topk_idx: torch.Tensor,
    rank: int,
    world_size: int,
    num_experts: int,
) -> RouteHandle:
    tokens, topk = topk_idx.shape
    local_experts = num_experts // world_size
    flat_experts = topk_idx.flatten()
    destinations = flat_experts // local_experts
    flat_order = torch.argsort(destinations, stable=True)
    send_counts = torch.bincount(
        destinations[flat_order], minlength=world_size
    ).to(torch.int64)
    send_splits, recv_splits = exchange_counts(send_counts)

    token_ids = torch.arange(tokens, device=topk_idx.device, dtype=torch.int64)
    slot_ids = torch.arange(topk, device=topk_idx.device, dtype=torch.int64)
    flat_tokens = token_ids[:, None].expand(-1, topk).reshape(-1)
    flat_slots = slot_ids[None, :].expand(tokens, -1).reshape(-1)
    token_order = flat_tokens[flat_order]
    send_metadata = torch.stack(
        (
            torch.full_like(token_order, rank),
            token_order,
            flat_slots[flat_order],
            flat_experts[flat_order],
        ),
        dim=1,
    )
    recv_metadata = exchange_tensor(send_metadata, send_splits, recv_splits)

    local_ids = recv_metadata[:, 3] - rank * local_experts
    expert_order = torch.argsort(local_ids, stable=True)
    grouped_metadata = recv_metadata[expert_order].contiguous()
    grouped_local_ids = local_ids[expert_order]
    seqlens = torch.bincount(
        grouped_local_ids, minlength=local_experts
    ).to(torch.int32)
    cu_seqlens = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=topk_idx.device),
            torch.cumsum(seqlens, dim=0, dtype=torch.int32),
        )
    )
    seqlens_list = seqlens.cpu().tolist()

    return_destinations = grouped_metadata[:, 0]
    return_order = torch.argsort(return_destinations, stable=True)
    return_send_counts = torch.bincount(
        return_destinations[return_order], minlength=world_size
    ).to(torch.int64)
    return_send_splits, return_recv_splits = exchange_counts(return_send_counts)
    returned_metadata = exchange_tensor(
        grouped_metadata[return_order, 1:4].contiguous(),
        return_send_splits,
        return_recv_splits,
    )

    return RouteHandle(
        token_order=token_order,
        flat_order=flat_order,
        send_splits=send_splits,
        recv_splits=recv_splits,
        expert_order=expert_order,
        grouped_metadata=grouped_metadata,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        seqlens_list=seqlens_list,
        return_order=return_order,
        return_send_splits=return_send_splits,
        return_recv_splits=return_recv_splits,
        returned_metadata=returned_metadata,
    )


def align_128(value: int) -> int:
    return max(128, ((value + 127) // 128) * 128)


class Nvfp4EPRuntime:
    def __init__(
        self,
        handle: RouteHandle,
        tokens: int,
        hidden: int,
        intermediate: int,
        total_experts: int,
        rank: int,
    ) -> None:
        self.handle = handle
        self.tokens = tokens
        self.hidden = hidden
        self.intermediate = intermediate
        self.total_experts = total_experts
        self.rank = rank
        self.local_experts = total_experts // 2
        self.rows = len(handle.grouped_metadata)
        device = handle.flat_order.device

        send_rows = len(handle.flat_order)
        return_rows = len(handle.returned_metadata)
        self.send_x = torch.empty(
            send_rows, hidden, dtype=torch.bfloat16, device=device
        )
        self.send_weight = torch.empty(
            send_rows, dtype=torch.float32, device=device
        )
        self.recv_x = torch.empty(
            self.rows, hidden, dtype=torch.bfloat16, device=device
        )
        self.recv_weight = torch.empty(
            self.rows, dtype=torch.float32, device=device
        )
        self.grouped_x = torch.empty_like(self.recv_x)
        self.grouped_weight = torch.empty_like(self.recv_weight)
        self.grouped_weight_bf16 = torch.empty(
            self.rows, dtype=torch.bfloat16, device=device
        )

        self.expert_scale_m_pad = align_128(max(handle.seqlens_list))
        self.global_scale_m_pad = align_128(self.rows)
        scale_k_hidden = sm120_nvfp4.scale_k_padded(hidden)
        self.direct_packed = torch.empty(
            self.rows, hidden // 2, dtype=torch.uint8, device=device
        )
        self.direct_scale = torch.empty(
            self.local_experts,
            self.expert_scale_m_pad * scale_k_hidden,
            dtype=torch.uint8,
            device=device,
        )
        self.baseline_packed = torch.empty_like(self.direct_packed)
        self.baseline_scale = torch.empty(
            1,
            self.global_scale_m_pad * scale_k_hidden,
            dtype=torch.uint8,
            device=device,
        )

        self.gate_up_weight = torch.full(
            (self.local_experts, 2 * intermediate, hidden // 2),
            0x11,
            dtype=torch.uint8,
            device=device,
        )
        self.down_weight = torch.full(
            (self.local_experts, hidden, intermediate // 2),
            0x11,
            dtype=torch.uint8,
            device=device,
        )
        self.gate_up_scale = torch.empty(
            self.local_experts,
            align_128(2 * intermediate) * scale_k_hidden,
            dtype=torch.uint8,
            device=device,
        )
        self.down_scale = torch.empty(
            self.local_experts,
            align_128(hidden) * sm120_nvfp4.scale_k_padded(intermediate),
            dtype=torch.uint8,
            device=device,
        )
        for local_id in range(self.local_experts):
            global_id = rank * self.local_experts + local_id
            self.gate_up_scale[local_id].fill_(0x10 + (global_id % 3) * 4)
            self.down_scale[local_id].fill_(0x10)

        workspace_bytes = sm120_nvfp4.expert_moe_workspace_bytes(
            self.rows,
            hidden,
            intermediate,
            self.local_experts,
            self.expert_scale_m_pad,
        )
        self.workspace = torch.empty(
            workspace_bytes, dtype=torch.uint8, device=device
        )
        self.direct_output = torch.empty(
            self.rows, hidden, dtype=torch.float16, device=device
        )
        self.baseline_output = torch.empty_like(self.direct_output)

        self.one_seqlen = torch.tensor(
            [self.rows], dtype=torch.int32, device=device
        )
        self.one_cu_seqlen = torch.tensor(
            [0, self.rows], dtype=torch.int32, device=device
        )
        self.baseline_topk_ids = (
            handle.grouped_metadata[:, 3].to(torch.int32).view(-1, 1)
        )
        self.baseline_topk_weight = torch.ones(
            self.rows, 1, dtype=torch.float32, device=device
        )

        self.weighted_output = torch.empty(
            self.rows, hidden, dtype=torch.bfloat16, device=device
        )
        self.return_send = torch.empty_like(self.weighted_output)
        self.return_recv = torch.empty(
            return_rows, hidden, dtype=torch.bfloat16, device=device
        )
        self.return_recv_f32 = torch.empty(
            return_rows, hidden, dtype=torch.float32, device=device
        )
        self.combined = torch.empty(
            tokens, hidden, dtype=torch.float32, device=device
        )

        self.comm_stream = torch.cuda.Stream(device=device)
        self.input_ready = torch.cuda.Event()
        self.dispatch_done = torch.cuda.Event()
        self.expert_ready = torch.cuda.Event()
        self.combine_done = torch.cuda.Event()

    def buffer_bytes(self) -> int:
        tensors = (
            self.send_x,
            self.send_weight,
            self.recv_x,
            self.recv_weight,
            self.grouped_x,
            self.grouped_weight,
            self.direct_packed,
            self.direct_scale,
            self.baseline_packed,
            self.baseline_scale,
            self.workspace,
            self.direct_output,
            self.baseline_output,
            self.weighted_output,
            self.return_send,
            self.return_recv,
            self.return_recv_f32,
            self.combined,
        )
        return sum(t.numel() * t.element_size() for t in tensors)

    def dispatch(self, x: torch.Tensor, topk_weight: torch.Tensor) -> None:
        handle = self.handle
        torch.index_select(x, 0, handle.token_order, out=self.send_x)
        torch.index_select(
            topk_weight.flatten(), 0, handle.flat_order, out=self.send_weight
        )
        compute_stream = torch.cuda.current_stream(x.device)
        self.input_ready.record(compute_stream)
        self.comm_stream.wait_event(self.input_ready)
        with torch.cuda.stream(self.comm_stream):
            exchange_tensor(
                self.send_x, handle.send_splits, handle.recv_splits, self.recv_x
            )
            exchange_tensor(
                self.send_weight,
                handle.send_splits,
                handle.recv_splits,
                self.recv_weight,
            )
            self.dispatch_done.record(self.comm_stream)

        compute_stream.wait_event(self.dispatch_done)
        torch.index_select(
            self.recv_x, 0, handle.expert_order, out=self.grouped_x
        )
        torch.index_select(
            self.recv_weight, 0, handle.expert_order, out=self.grouped_weight
        )
        self.grouped_weight_bf16.copy_(self.grouped_weight)

    def compute(self, path: str) -> torch.Tensor:
        if path == "direct":
            sm120_nvfp4.quantize_expert(
                self.grouped_x,
                self.handle.seqlens,
                self.handle.cu_seqlens,
                scale_m_pad=self.expert_scale_m_pad,
                output=self.direct_packed,
                output_scale=self.direct_scale,
            )
            return sm120_nvfp4.expert_moe(
                self.direct_packed,
                self.direct_scale,
                self.gate_up_weight,
                self.gate_up_scale,
                self.down_weight,
                self.down_scale,
                self.handle.seqlens,
                self.handle.cu_seqlens,
                output=self.direct_output,
                workspace=self.workspace,
            )

        if path == "reroute":
            sm120_nvfp4.quantize_expert(
                self.grouped_x,
                self.one_seqlen,
                self.one_cu_seqlen,
                scale_m_pad=self.global_scale_m_pad,
                output=self.baseline_packed,
                output_scale=self.baseline_scale,
            )
            return sm120_nvfp4.fused_moe(
                self.baseline_packed,
                self.baseline_scale.flatten(),
                self.gate_up_weight,
                self.gate_up_scale,
                self.down_weight,
                self.down_scale,
                self.baseline_topk_ids,
                self.baseline_topk_weight,
                ep_rank=self.rank,
                num_experts=self.total_experts,
                output=self.baseline_output,
            )
        raise ValueError(f"unknown path: {path}")

    def combine(self, expert_output: torch.Tensor) -> torch.Tensor:
        handle = self.handle
        self.weighted_output.copy_(expert_output)
        self.weighted_output.mul_(self.grouped_weight_bf16.unsqueeze(1))
        torch.index_select(
            self.weighted_output, 0, handle.return_order, out=self.return_send
        )

        compute_stream = torch.cuda.current_stream(expert_output.device)
        self.expert_ready.record(compute_stream)
        self.comm_stream.wait_event(self.expert_ready)
        with torch.cuda.stream(self.comm_stream):
            exchange_tensor(
                self.return_send,
                handle.return_send_splits,
                handle.return_recv_splits,
                self.return_recv,
            )
            self.combine_done.record(self.comm_stream)

        compute_stream.wait_event(self.combine_done)
        self.return_recv_f32.copy_(self.return_recv)
        self.combined.zero_()
        self.combined.index_add_(
            0, handle.returned_metadata[:, 0], self.return_recv_f32
        )
        return self.combined

    def forward(
        self, x: torch.Tensor, topk_weight: torch.Tensor, path: str
    ) -> torch.Tensor:
        self.dispatch(x, topk_weight)
        return self.combine(self.compute(path))


def percentile(samples: list[float], p: float) -> float:
    ordered = sorted(samples)
    index = max(
        0, min(len(ordered) - 1, int(len(ordered) * p + 0.999999) - 1)
    )
    return ordered[index]


def measure_pair_rank_max_ms(
    first: Callable[[], object],
    second: Callable[[], object],
    iterations: int,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    first_samples: list[float] = []
    second_samples: list[float] = []

    def measure_one(function: Callable[[], object]) -> float:
        dist.barrier()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        function()
        torch.cuda.synchronize(device)
        elapsed = torch.tensor(
            (time.perf_counter() - started) * 1000.0,
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        return elapsed.item()

    for iteration in range(iterations):
        if iteration % 2 == 0:
            first_samples.append(measure_one(first))
            second_samples.append(measure_one(second))
        else:
            second_samples.append(measure_one(second))
            first_samples.append(measure_one(first))
    return first_samples, second_samples


def run(args: argparse.Namespace) -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError("benchmark is guarded to exactly two GPUs")
    if args.experts % world_size != 0:
        raise ValueError("experts must divide evenly across two ranks")

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(20260911 + rank)

    x = (
        torch.randn(
            args.tokens, args.hidden, dtype=torch.bfloat16, device=device
        )
        * args.input_scale
    ).contiguous()
    topk_idx = make_topk(
        args.tokens,
        args.topk,
        args.experts,
        rank,
        args.routing,
        device,
    )
    base_weight = torch.arange(
        1, args.topk + 1, dtype=torch.float32, device=device
    )
    base_weight /= base_weight.sum()
    topk_weight = base_weight.expand(args.tokens, -1).contiguous()

    handle = build_route_handle(
        topk_idx, rank, world_size, args.experts
    )
    runtime = Nvfp4EPRuntime(
        handle,
        args.tokens,
        args.hidden,
        args.intermediate,
        args.experts,
        rank,
    )

    for _ in range(args.warmup):
        runtime.forward(x, topk_weight, "direct")
        runtime.forward(x, topk_weight, "reroute")
    torch.cuda.synchronize(device)

    direct = runtime.forward(x, topk_weight, "direct").clone()
    reroute = runtime.forward(x, topk_weight, "reroute").clone()
    torch.cuda.synchronize(device)
    error = (direct - reroute).abs().max()
    dist.all_reduce(error, op=dist.ReduceOp.MAX)

    runtime.dispatch(x, topk_weight)
    torch.cuda.synchronize(device)
    direct_compute, reroute_compute = measure_pair_rank_max_ms(
        lambda: runtime.compute("direct"),
        lambda: runtime.compute("reroute"),
        args.iterations,
        device,
    )
    direct_e2e, reroute_e2e = measure_pair_rank_max_ms(
        lambda: runtime.forward(x, topk_weight, "direct"),
        lambda: runtime.forward(x, topk_weight, "reroute"),
        args.iterations,
        device,
    )

    local_counts = torch.tensor(
        handle.seqlens_list, dtype=torch.int64, device=device
    )
    max_count = local_counts.max()
    min_count = local_counts.min()
    dist.all_reduce(max_count, op=dist.ReduceOp.MAX)
    dist.all_reduce(min_count, op=dist.ReduceOp.MIN)

    if rank == 0:
        direct_compute_p50 = percentile(direct_compute, 0.50)
        reroute_compute_p50 = percentile(reroute_compute, 0.50)
        direct_e2e_p50 = percentile(direct_e2e, 0.50)
        reroute_e2e_p50 = percentile(reroute_e2e, 0.50)
        result = {
            "status": "PASS" if error.item() == 0.0 else "FAIL",
            "scope": (
                "DeepEP-compatible torch.distributed reference transport; "
                "not native DeepEP bandwidth"
            ),
            "physical_gpu_policy": (
                "launch with CUDA_VISIBLE_DEVICES=1,2; exactly two ranks"
            ),
            "routing": args.routing,
            "tokens_per_rank": args.tokens,
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "topk": args.topk,
            "experts": args.experts,
            "max_abs_diff_direct_vs_reroute": error.item(),
            "expert_tokens_global_min": min_count.item(),
            "expert_tokens_global_max": max_count.item(),
            "direct_compute_p50_ms": direct_compute_p50,
            "direct_compute_p95_ms": percentile(direct_compute, 0.95),
            "reroute_compute_p50_ms": reroute_compute_p50,
            "reroute_compute_p95_ms": percentile(reroute_compute, 0.95),
            "compute_speedup": reroute_compute_p50 / direct_compute_p50,
            "direct_e2e_p50_ms": direct_e2e_p50,
            "direct_e2e_p95_ms": percentile(direct_e2e, 0.95),
            "reroute_e2e_p50_ms": reroute_e2e_p50,
            "reroute_e2e_p95_ms": percentile(reroute_e2e, 0.95),
            "e2e_speedup": reroute_e2e_p50 / direct_e2e_p50,
            "route_handle_bytes_rank0": handle.bytes(),
            "preallocated_buffer_bytes_rank0": runtime.buffer_bytes(),
            "direct_path": (
                "expanded expert-major layout -> BF16/NVFP4 quantize -> "
                "expert_moe reusable workspace -> combine"
            ),
            "reroute_path": (
                "same expanded layout -> global BF16/NVFP4 quantize -> "
                "fused_moe repeats local count/gather/reduce and allocates "
                "temporary tensors -> combine"
            ),
        }
        print(json.dumps(result, indent=2, sort_keys=True))

    dist.barrier()
    dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument(
        "--routing", choices=("balanced", "skewed"), default="balanced"
    )
    parser.add_argument("--input-scale", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
