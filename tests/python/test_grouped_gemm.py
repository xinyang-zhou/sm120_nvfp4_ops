import unittest

import torch

import sm120_nvfp4


def require_sm120() -> None:
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (12, 0):
        raise unittest.SkipTest(f"SM120 is required, got SM{major}{minor}")


class GroupedGemmTest(unittest.TestCase):
    def test_uniform_scales(self) -> None:
        require_sm120()
        device = "cuda"
        groups, total_m, n, k = 2, 5, 128, 128
        seqlens = torch.tensor([2, 3], dtype=torch.int32, device=device)
        cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32, device=device)
        x = torch.full(
            (total_m, k // 2), 0x22, dtype=torch.uint8, device=device
        )
        weight = torch.full(
            (groups, n, k // 2), 0x22, dtype=torch.uint8, device=device
        )
        x_scale = torch.full(
            (groups, 128 * 8), 0x38, dtype=torch.uint8, device=device
        )
        weight_scale = torch.full(
            (groups, 128 * 8), 0x38, dtype=torch.uint8, device=device
        )

        output = sm120_nvfp4.grouped_gemm(
            x, weight, seqlens, cu_seqlens, x_scale, weight_scale
        )
        torch.cuda.synchronize()

        self.assertEqual(output.shape, (total_m, n))
        self.assertEqual(output.dtype, torch.float16)
        torch.testing.assert_close(
            output.float(),
            torch.full_like(output.float(), 128.0),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
