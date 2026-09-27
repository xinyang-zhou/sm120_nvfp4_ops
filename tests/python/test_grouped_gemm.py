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
    def test_empty_experts_tile_tails_and_changed_metadata(self) -> None:
        require_sm120()
        torch.manual_seed(921)
        groups, rows, n, k, pad = 4, 257, 128, 128, 256
        x = torch.randint(256, (rows, k // 2), dtype=torch.uint8, device="cuda")
        weight = torch.randint(256, (groups, n, k // 2), dtype=torch.uint8, device="cuda")
        codes = torch.tensor([0x10, 0x18, 0x20, 0x28], device="cuda", dtype=torch.uint8)
        xs = codes[:, None].expand(-1, pad * sm120_nvfp4.scale_k_padded(k)).contiguous()
        ws = codes.flip(0)[:, None].expand(-1, sm120_nvfp4.scale_b_elements(1, n, k)).contiguous()
        levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                               -0., -.5, -1., -1.5, -2., -3., -4., -6.], device="cuda")

        def decode(packed):
            indices = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
            return levels[indices]

        lengths = torch.empty(groups, device="cuda", dtype=torch.int32)
        offsets = torch.empty(groups + 1, device="cuda", dtype=torch.int32)
        output = torch.empty((rows, n), device="cuda", dtype=torch.float16)
        for counts in ([0, 1, 127, 129], [129, 0, 1, 127]):
            prefix = [0]
            for count in counts:
                prefix.append(prefix[-1] + count)
            lengths.copy_(torch.tensor(counts, device="cuda", dtype=torch.int32))
            offsets.copy_(torch.tensor(prefix, device="cuda", dtype=torch.int32))
            expected = torch.empty_like(output)
            for group in range(groups):
                begin, end = prefix[group:group + 2]
                a = decode(x[begin:end]) * codes[group:group + 1].view(torch.float8_e4m3fn).float()
                b = decode(weight[group]) * codes[3 - group:4 - group].view(torch.float8_e4m3fn).float()
                expected[begin:end] = (a @ b.T).half()
            output.fill_(float("nan"))
            actual = sm120_nvfp4.grouped_gemm(x, weight, lengths, offsets, xs, ws, output=output)
            self.assertEqual(actual.data_ptr(), output.data_ptr())
            # Power-of-two scales keep these finite dyadic dot products exact.
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

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
