import unittest

import torch

import sm120_nvfp4


def require_sm120() -> None:
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (12, 0):
        raise unittest.SkipTest(f"SM120 is required, got SM{major}{minor}")


def make_case():
    device = "cuda"
    tokens, experts, hidden, intermediate = 4, 2, 128, 64
    x = torch.full(
        (tokens, hidden // 2), 0x11, dtype=torch.uint8, device=device
    )
    gate_up_weight = torch.full(
        (experts, 2 * intermediate, hidden // 2),
        0x11,
        dtype=torch.uint8,
        device=device,
    )
    down_weight = torch.full(
        (experts, hidden, intermediate // 2),
        0x11,
        dtype=torch.uint8,
        device=device,
    )
    x_scale = torch.full((128 * 8,), 0x38, dtype=torch.uint8, device=device)
    gate_up_scale = torch.full(
        (experts, 128 * 8), 0x38, dtype=torch.uint8, device=device
    )
    down_scale = torch.full(
        (experts, 128 * 4), 0x38, dtype=torch.uint8, device=device
    )
    return (
        tokens,
        experts,
        hidden,
        x,
        x_scale,
        gate_up_weight,
        gate_up_scale,
        down_weight,
        down_scale,
    )


class FusedMoeTest(unittest.TestCase):
    def test_uniform_data(self) -> None:
        require_sm120()
        (
            tokens,
            experts,
            hidden,
            x,
            x_scale,
            gate_up_weight,
            gate_up_scale,
            down_weight,
            down_scale,
        ) = make_case()
        topk_ids = torch.tensor(
            [[0, 1], [1, 0], [0, 1], [1, 0]], dtype=torch.int32, device="cuda"
        )
        topk_weights = torch.tensor(
            [[0.25, 0.75], [0.6, 0.4], [0.5, 0.5], [0.2, 0.8]],
            dtype=torch.float32,
            device="cuda",
        )
        shared_output = torch.full(
            (tokens, hidden), 2.0, dtype=torch.float16, device="cuda"
        )
        output = torch.empty_like(shared_output)

        result = sm120_nvfp4.fused_moe(
            x,
            x_scale,
            gate_up_weight,
            gate_up_scale,
            down_weight,
            down_scale,
            topk_ids,
            topk_weights,
            shared_output=shared_output,
            output=output,
        )
        torch.cuda.synchronize()

        self.assertEqual(result.data_ptr(), output.data_ptr())
        torch.testing.assert_close(
            result.float(),
            torch.full_like(result.float(), 33792.0),
            rtol=0,
            atol=0,
        )

    def test_nonlocal_routes_are_ignored(self) -> None:
        require_sm120()
        (
            tokens,
            _,
            hidden,
            x,
            x_scale,
            gate_up_weight,
            gate_up_scale,
            down_weight,
            down_scale,
        ) = make_case()
        remote_ids = torch.full(
            (tokens, 1), 2, dtype=torch.int32, device="cuda"
        )
        topk_weights = torch.ones(
            (tokens, 1), dtype=torch.float32, device="cuda"
        )
        shared_output = torch.full(
            (tokens, hidden), 2.0, dtype=torch.float16, device="cuda"
        )

        result = sm120_nvfp4.fused_moe(
            x,
            x_scale,
            gate_up_weight,
            gate_up_scale,
            down_weight,
            down_scale,
            remote_ids,
            topk_weights,
            shared_output=shared_output,
            ep_rank=0,
            num_experts=4,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(result, shared_output, rtol=0, atol=0)

    def test_nonuniform_activation_scales_survive_routing(self) -> None:
        require_sm120()
        (
            tokens,
            experts,
            _,
            x,
            x_scale,
            gate_up_weight,
            gate_up_scale,
            down_weight,
            down_scale,
        ) = make_case()

        # For M=K=128, address the first four logical rows through the CUTLASS
        # M(32x4)xK(4xrest_k) physical SFA layout. Alternating 0.5/1.0 scales
        # verify that routing copies each token scale instead of only its data.
        nonuniform_scale = torch.zeros_like(x_scale)
        for row, code in enumerate((0x30, 0x38, 0x30, 0x38)):
            for k_block in range(8):
                offset = (
                    (k_block // 4) * 512
                    + (row % 32) * 16
                    + (row // 32) * 4
                    + (k_block % 4)
                )
                nonuniform_scale[offset] = code

        topk_ids = torch.tensor(
            [[1], [0], [1], [0]], dtype=torch.int32, device="cuda"
        )
        topk_weights = torch.ones(
            (tokens, 1), dtype=torch.float32, device="cuda"
        )
        result = sm120_nvfp4.fused_moe(
            x,
            nonuniform_scale,
            gate_up_weight,
            gate_up_scale,
            down_weight,
            down_scale,
            topk_ids,
            topk_weights,
            num_experts=experts,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(
            result[:, 0].float(),
            torch.tensor(
                [8448.0, 33792.0, 8448.0, 33792.0], device="cuda"
            ),
            rtol=0,
            atol=0,
        )


    def test_expert_major_handoff_matches_reroute(self) -> None:
        require_sm120()
        device = "cuda"
        rows, experts, hidden, intermediate = 5, 2, 128, 64
        torch.manual_seed(7)
        x = (
            torch.randn(rows, hidden, dtype=torch.bfloat16, device=device)
            * 0.25
        ).contiguous()
        seqlens = torch.tensor(
            [2, 3], dtype=torch.int32, device=device
        )
        cu_seqlens = torch.tensor(
            [0, 2, 5], dtype=torch.int32, device=device
        )
        packed, scales = sm120_nvfp4.quantize_expert(
            x, seqlens, cu_seqlens, scale_m_pad=128
        )

        gate_up_weight = torch.full(
            (experts, 2 * intermediate, hidden // 2),
            0x11,
            dtype=torch.uint8,
            device=device,
        )
        down_weight = torch.full(
            (experts, hidden, intermediate // 2),
            0x11,
            dtype=torch.uint8,
            device=device,
        )
        gate_up_scale = torch.full(
            (
                experts,
                128 * sm120_nvfp4.scale_k_padded(hidden),
            ),
            0x38,
            dtype=torch.uint8,
            device=device,
        )
        down_scale = torch.full(
            (
                experts,
                128 * sm120_nvfp4.scale_k_padded(intermediate),
            ),
            0x38,
            dtype=torch.uint8,
            device=device,
        )

        workspace_bytes = sm120_nvfp4.expert_moe_workspace_bytes(
            rows, hidden, intermediate, experts, 128
        )
        workspace = torch.empty(
            workspace_bytes, dtype=torch.uint8, device=device
        )
        direct_output = torch.empty(
            rows, hidden, dtype=torch.float16, device=device
        )
        direct = sm120_nvfp4.expert_moe(
            packed,
            scales,
            gate_up_weight,
            gate_up_scale,
            down_weight,
            down_scale,
            seqlens,
            cu_seqlens,
            output=direct_output,
            workspace=workspace,
        )

        one_seqlen = torch.tensor(
            [rows], dtype=torch.int32, device=device
        )
        one_cu_seqlen = torch.tensor(
            [0, rows], dtype=torch.int32, device=device
        )
        baseline_input, baseline_scale = sm120_nvfp4.quantize_expert(
            x, one_seqlen, one_cu_seqlen, scale_m_pad=128
        )
        topk_ids = torch.tensor(
            [[0], [0], [1], [1], [1]],
            dtype=torch.int32,
            device=device,
        )
        topk_weights = torch.ones(
            rows, 1, dtype=torch.float32, device=device
        )
        baseline = sm120_nvfp4.fused_moe(
            baseline_input,
            baseline_scale.flatten(),
            gate_up_weight,
            gate_up_scale,
            down_weight,
            down_scale,
            topk_ids,
            topk_weights,
        )
        torch.cuda.synchronize()

        self.assertEqual(direct.data_ptr(), direct_output.data_ptr())
        torch.testing.assert_close(direct, baseline, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
