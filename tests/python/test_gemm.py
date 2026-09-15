import unittest

import torch

import sm120_nvfp4


def require_sm120() -> None:
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (12, 0):
        raise unittest.SkipTest(f"SM120 is required, got SM{major}{minor}")


class GemmTest(unittest.TestCase):
    def test_cute_matches_cutlass_reference(self) -> None:
        require_sm120()
        m, n, k = 16, 128, 128
        # E2M1 1.0 is nibble 0x2; UE4M3 1.0 is byte 0x38.
        a = torch.full((m, k // 2), 0x22, dtype=torch.uint8, device="cuda")
        b = torch.full((n, k // 2), 0x22, dtype=torch.uint8, device="cuda")
        sfa = torch.full(
            (sm120_nvfp4.scale_a_elements(m, n, k),),
            0x38,
            dtype=torch.uint8,
            device="cuda",
        )
        sfb = torch.full(
            (sm120_nvfp4.scale_b_elements(m, n, k),),
            0x38,
            dtype=torch.uint8,
            device="cuda",
        )

        cute_output = sm120_nvfp4.cute_gemm(a, b, sfa, sfb)
        default_output = sm120_nvfp4.gemm(a, b, sfa, sfb)
        cutlass_output = sm120_nvfp4.cutlass_gemm(a, b, sfa, sfb)
        torch.cuda.synchronize()

        expected = torch.full(
            (m, n), float(k), dtype=torch.float16, device="cuda"
        )
        torch.testing.assert_close(cute_output, expected, rtol=0, atol=0)
        torch.testing.assert_close(default_output, cute_output, rtol=0, atol=0)
        torch.testing.assert_close(cutlass_output, cute_output, rtol=0, atol=0)

    def test_tma_epilogue_partial_tile(self) -> None:
        require_sm120()
        # M=64 selects TMA for a partial tile. M=129 additionally exercises an
        # extreme one-row TMA tail through the TensorMap OOB predicate.
        n, k = 136, 128
        for m in (64, 129):
            with self.subTest(m=m):
                a = torch.full(
                    (m, k // 2), 0x22, dtype=torch.uint8, device="cuda"
                )
                b = torch.full(
                    (n, k // 2), 0x22, dtype=torch.uint8, device="cuda"
                )
                sfa = torch.full(
                    (sm120_nvfp4.scale_a_elements(m, n, k),),
                    0x38,
                    dtype=torch.uint8,
                    device="cuda",
                )
                sfb = torch.full(
                    (sm120_nvfp4.scale_b_elements(m, n, k),),
                    0x38,
                    dtype=torch.uint8,
                    device="cuda",
                )

                output = sm120_nvfp4.cute_gemm(a, b, sfa, sfb)
                torch.cuda.synchronize()

                expected = torch.full(
                    (m, n), float(k), dtype=torch.float16, device="cuda"
                )
                torch.testing.assert_close(output, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
