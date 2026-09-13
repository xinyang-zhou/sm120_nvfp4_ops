import unittest

import torch

import sm120_nvfp4


def require_sm120() -> None:
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (12, 0):
        raise unittest.SkipTest(f"SM120 is required, got SM{major}{minor}")


class AttentionTest(unittest.TestCase):
    @staticmethod
    def pack_codes(codes: torch.Tensor) -> torch.Tensor:
        return (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)

    def make_problem(self):
        batch, heads = 1, 1
        query_length, kv_length = 32, 128
        head_dim, value_dim = 128, 128

        # Uniform Q/K makes every visible score equal. V is one for the first
        # 64 KV positions and zero for the final 64 positions.
        query = torch.full(
            (batch, heads, query_length, head_dim // 2),
            0x22,
            dtype=torch.uint8,
            device="cuda",
        )
        key = torch.full(
            (batch, heads, kv_length, head_dim // 2),
            0x22,
            dtype=torch.uint8,
            device="cuda",
        )
        value_transposed = torch.zeros(
            (batch, heads, value_dim, kv_length // 2),
            dtype=torch.uint8,
            device="cuda",
        )
        value_transposed[..., : kv_length // 4] = 0x22

        matrices = batch * heads
        query_scale = torch.full(
            (
                matrices
                * sm120_nvfp4.scale_a_elements(query_length, kv_length, head_dim),
            ),
            0x38,
            dtype=torch.uint8,
            device="cuda",
        )
        key_scale = torch.full(
            (
                matrices
                * sm120_nvfp4.scale_b_elements(query_length, kv_length, head_dim),
            ),
            0x38,
            dtype=torch.uint8,
            device="cuda",
        )
        value_scale = torch.full(
            (
                matrices
                * sm120_nvfp4.scale_b_elements(query_length, value_dim, kv_length),
            ),
            0x38,
            dtype=torch.uint8,
            device="cuda",
        )
        return (
            query,
            key,
            value_transposed,
            query_scale,
            key_scale,
            value_scale,
        )

    def test_noncausal_and_reusable_buffers(self) -> None:
        require_sm120()
        problem = self.make_problem()
        query = problem[0]
        batch, heads, query_length = query.shape[:3]
        kv_length = problem[1].shape[2]
        value_dim = problem[2].shape[2]
        workspace = torch.empty(
            sm120_nvfp4.attention_workspace_bytes(
                batch, heads, query_length, kv_length
            ),
            dtype=torch.uint8,
            device="cuda",
        )
        output = torch.empty(
            (batch, heads, query_length, value_dim),
            dtype=torch.float16,
            device="cuda",
        )
        result = sm120_nvfp4.attention_prefill(
            *problem, causal=False, output=output, workspace=workspace
        )
        torch.cuda.synchronize()

        self.assertEqual(result.data_ptr(), output.data_ptr())
        expected = torch.full_like(result, 0.5)
        torch.testing.assert_close(result, expected, rtol=0, atol=0.08)

    def test_suffix_aligned_causal_mask(self) -> None:
        require_sm120()
        problem = self.make_problem()
        result = sm120_nvfp4.attention_prefill(*problem, causal=True)
        torch.cuda.synchronize()

        # M=32,N=128 maps query row r to KV position 96+r. With uniform
        # scores, the expected average is 64 / (97+r).
        rows = torch.arange(32, dtype=torch.float32, device="cuda")
        expected_rows = 64.0 / (97.0 + rows)
        expected = expected_rows.view(1, 1, 32, 1).expand_as(result).to(result.dtype)
        torch.testing.assert_close(result, expected, rtol=0, atol=0.08)

    def test_nonuniform_matches_fp32_reference(self) -> None:
        require_sm120()
        torch.manual_seed(7)
        batch, heads = 1, 2
        query_length, kv_length = 16, 32
        head_dim, value_dim = 32, 8

        # Limit Q/K to +/-0.5 E2M1 payloads, then apply a 0.125 UE4M3
        # scale. This keeps logits in a range that exercises the full
        # softmax distribution instead of collapsing it to one-hot.
        qk_codebook = torch.tensor([0x1, 0x9], dtype=torch.uint8, device="cuda")
        q_codes = qk_codebook[
            torch.randint(0, 2, (batch, heads, query_length, head_dim), device="cuda")
        ]
        k_codes = qk_codebook[
            torch.randint(0, 2, (batch, heads, kv_length, head_dim), device="cuda")
        ]
        v_codebook = torch.tensor([0x0, 0x2, 0xA], dtype=torch.uint8, device="cuda")
        v_codes = v_codebook[
            torch.randint(0, 3, (batch, heads, value_dim, kv_length), device="cuda")
        ]

        query = self.pack_codes(q_codes)
        key = self.pack_codes(k_codes)
        value_transposed = self.pack_codes(v_codes)
        matrices = batch * heads
        query_scale = torch.full(
            (
                matrices
                * sm120_nvfp4.scale_a_elements(query_length, kv_length, head_dim),
            ),
            0x20,  # UE4M3 0.125
            dtype=torch.uint8,
            device="cuda",
        )
        key_scale = torch.full(
            (
                matrices
                * sm120_nvfp4.scale_b_elements(query_length, kv_length, head_dim),
            ),
            0x20,
            dtype=torch.uint8,
            device="cuda",
        )
        value_scale = torch.full(
            (
                matrices
                * sm120_nvfp4.scale_b_elements(query_length, value_dim, kv_length),
            ),
            0x30,  # UE4M3 0.5
            dtype=torch.uint8,
            device="cuda",
        )

        softmax_scale = head_dim**-0.5
        result = sm120_nvfp4.attention_prefill(
            query,
            key,
            value_transposed,
            query_scale,
            key_scale,
            value_scale,
            causal=False,
            softmax_scale=softmax_scale,
        )

        decode = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            dtype=torch.float32,
            device="cuda",
        )
        q_reference = decode[q_codes.long()] * 0.125
        k_reference = decode[k_codes.long()] * 0.125
        v_reference = decode[v_codes.long()] * 0.5
        scores = (
            torch.einsum("bhmd,bhnd->bhmn", q_reference, k_reference) * softmax_scale
        )
        expected = torch.einsum(
            "bhmn,bhdn->bhmd", scores.softmax(dim=-1), v_reference
        ).to(torch.float16)
        torch.cuda.synchronize()

        torch.testing.assert_close(result, expected, rtol=0.25, atol=0.05)

    def test_decode_gqa_with_dynamic_lengths_and_reusable_buffers(self) -> None:
        require_sm120()
        batch, query_heads, kv_heads = 2, 4, 2
        max_kv_length, head_dim, value_dim = 32, 32, 8

        query = torch.full(
            (batch, query_heads, head_dim // 2),
            0x22,
            dtype=torch.uint8,
            device="cuda",
        )
        key_cache = torch.full(
            (batch, kv_heads, max_kv_length, head_dim // 2),
            0x22,
            dtype=torch.uint8,
            device="cuda",
        )
        value_codes = torch.zeros(
            (batch, kv_heads, value_dim, max_kv_length),
            dtype=torch.uint8,
            device="cuda",
        )
        value_codes[0, 0, :, :8] = 0x2
        value_codes[0, 1, :, :16] = 0x2
        value_codes[1, 0, :, :16] = 0x2
        value_codes[1, 1, :, :8] = 0x2
        value_cache_transposed = self.pack_codes(value_codes)

        query_matrices = batch * query_heads
        kv_matrices = batch * kv_heads
        query_scale = torch.full(
            (
                query_matrices
                * sm120_nvfp4.scale_a_elements(1, max_kv_length, head_dim),
            ),
            0x38,
            dtype=torch.uint8,
            device="cuda",
        )
        key_scale = torch.full(
            (kv_matrices * sm120_nvfp4.scale_b_elements(1, max_kv_length, head_dim),),
            0x38,
            dtype=torch.uint8,
            device="cuda",
        )
        value_scale = torch.full(
            (kv_matrices * sm120_nvfp4.scale_b_elements(1, value_dim, max_kv_length),),
            0x38,
            dtype=torch.uint8,
            device="cuda",
        )
        kv_lengths = torch.tensor([16, 32], dtype=torch.int32, device="cuda")
        output = torch.empty(
            (batch, query_heads, value_dim),
            dtype=torch.float16,
            device="cuda",
        )
        workspace = torch.empty(
            sm120_nvfp4.attention_decode_workspace_bytes(
                batch, query_heads, kv_heads, max_kv_length, head_dim, value_dim
            ),
            dtype=torch.uint8,
            device="cuda",
        )

        result = sm120_nvfp4.attention_decode(
            query,
            key_cache,
            value_cache_transposed,
            query_scale,
            key_scale,
            value_scale,
            kv_lengths=kv_lengths,
            output=output,
            workspace=workspace,
        )
        torch.cuda.synchronize()

        self.assertEqual(result.data_ptr(), output.data_ptr())
        expected_heads = torch.tensor(
            [[0.5, 0.5, 1.0, 1.0], [0.5, 0.5, 0.25, 0.25]],
            dtype=torch.float16,
            device="cuda",
        )
        expected = expected_heads.unsqueeze(-1).expand_as(result)
        torch.testing.assert_close(result, expected, rtol=0, atol=0.08)

    def test_decode_nonuniform_matches_fp32_reference(self) -> None:
        require_sm120()
        torch.manual_seed(11)
        batch, query_heads, kv_heads = 1, 4, 2
        # Exercise split-K, two QK K-tiles and two PV value-dimension tiles.
        max_kv_length, head_dim, value_dim = 256, 160, 136
        valid_length = 191

        qk_codebook = torch.tensor([0x1, 0x9], dtype=torch.uint8, device="cuda")
        query_codes = qk_codebook[
            torch.randint(0, 2, (batch, query_heads, head_dim), device="cuda")
        ]
        key_codes = qk_codebook[
            torch.randint(
                0,
                2,
                (batch, kv_heads, max_kv_length, head_dim),
                device="cuda",
            )
        ]
        value_codebook = torch.tensor([0x0, 0x2, 0xA], dtype=torch.uint8, device="cuda")
        value_codes = value_codebook[
            torch.randint(
                0,
                3,
                (batch, kv_heads, value_dim, max_kv_length),
                device="cuda",
            )
        ]
        query = self.pack_codes(query_codes)
        key_cache = self.pack_codes(key_codes)
        value_cache_transposed = self.pack_codes(value_codes)

        query_matrices = batch * query_heads
        kv_matrices = batch * kv_heads
        query_scale = torch.full(
            (
                query_matrices
                * sm120_nvfp4.scale_a_elements(1, max_kv_length, head_dim),
            ),
            0x20,
            dtype=torch.uint8,
            device="cuda",
        )
        key_scale = torch.full(
            (kv_matrices * sm120_nvfp4.scale_b_elements(1, max_kv_length, head_dim),),
            0x20,
            dtype=torch.uint8,
            device="cuda",
        )
        value_scale = torch.full(
            (kv_matrices * sm120_nvfp4.scale_b_elements(1, value_dim, max_kv_length),),
            0x30,
            dtype=torch.uint8,
            device="cuda",
        )
        softmax_scale = head_dim**-0.5
        result = sm120_nvfp4.attention_decode(
            query,
            key_cache,
            value_cache_transposed,
            query_scale,
            key_scale,
            value_scale,
            kv_lengths=torch.tensor(
                [valid_length], dtype=torch.int32, device="cuda"
            ),
            softmax_scale=softmax_scale,
        )

        decode = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            dtype=torch.float32,
            device="cuda",
        )
        query_reference = decode[query_codes.long()] * 0.125
        key_reference = decode[key_codes.long()] * 0.125
        value_reference = decode[value_codes.long()] * 0.5
        kv_head_index = torch.arange(query_heads, device="cuda") // (
            query_heads // kv_heads
        )
        expanded_key = key_reference[:, kv_head_index]
        expanded_value = value_reference[:, kv_head_index]
        scores = (
            torch.einsum(
                "bhd,bhnd->bhn",
                query_reference,
                expanded_key[..., :valid_length, :],
            )
            * softmax_scale
        )
        expected = torch.einsum(
            "bhn,bhdn->bhd",
            scores.softmax(dim=-1),
            expanded_value[..., :valid_length],
        ).to(torch.float16)
        torch.cuda.synchronize()

        torch.testing.assert_close(result, expected, rtol=0.25, atol=0.05)

    def test_paged_decode_matches_fp32_reference(self) -> None:
        require_sm120()
        batch, query_heads, kv_heads = 2, 4, 2
        head_dim, value_dim = 160, 136
        max_kv_length = 256
        valid_lengths = torch.tensor(
            [191, 129], dtype=torch.int32, device="cuda"
        )
        softmax_scale = head_dim**-0.5

        decode = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            dtype=torch.float32,
            device="cuda",
        )

        for block_size in (32, 64, 128):
            with self.subTest(block_size=block_size):
                torch.manual_seed(31 + block_size)
                max_blocks = max_kv_length // block_size
                physical_blocks = max_blocks + 3
                block_table = torch.stack(
                    [
                        torch.randperm(
                            physical_blocks,
                            dtype=torch.int32,
                            device="cuda",
                        )[:max_blocks]
                        for _ in range(batch)
                    ]
                ).contiguous()

                qk_codebook = torch.tensor(
                    [0x1, 0x9], dtype=torch.uint8, device="cuda"
                )
                query_codes = qk_codebook[
                    torch.randint(
                        0,
                        2,
                        (batch, query_heads, head_dim),
                        device="cuda",
                    )
                ]
                key_codes = qk_codebook[
                    torch.randint(
                        0,
                        2,
                        (physical_blocks, kv_heads, block_size, head_dim),
                        device="cuda",
                    )
                ]
                value_codebook = torch.tensor(
                    [0x0, 0x2, 0xA], dtype=torch.uint8, device="cuda"
                )
                value_codes = value_codebook[
                    torch.randint(
                        0,
                        3,
                        (physical_blocks, kv_heads, value_dim, block_size),
                        device="cuda",
                    )
                ]
                query = self.pack_codes(query_codes)
                key_cache = self.pack_codes(key_codes)
                value_cache_transposed = self.pack_codes(value_codes)

                query_scale = torch.full(
                    (
                        batch
                        * query_heads
                        * sm120_nvfp4.scale_a_elements(
                            1, max_kv_length, head_dim
                        ),
                    ),
                    0x20,
                    dtype=torch.uint8,
                    device="cuda",
                )
                key_scale = torch.full(
                    (
                        physical_blocks
                        * kv_heads
                        * sm120_nvfp4.scale_b_elements(
                            1, block_size, head_dim
                        ),
                    ),
                    0x20,
                    dtype=torch.uint8,
                    device="cuda",
                )
                value_scale = torch.full(
                    (
                        physical_blocks
                        * kv_heads
                        * sm120_nvfp4.scale_b_elements(
                            1, value_dim, block_size
                        ),
                    ),
                    0x30,
                    dtype=torch.uint8,
                    device="cuda",
                )
                workspace = torch.empty(
                    sm120_nvfp4.attention_decode_workspace_bytes(
                        batch,
                        query_heads,
                        kv_heads,
                        max_kv_length,
                        head_dim,
                        value_dim,
                    ),
                    dtype=torch.uint8,
                    device="cuda",
                )
                output = torch.empty(
                    (batch, query_heads, value_dim),
                    dtype=torch.float16,
                    device="cuda",
                )
                result = sm120_nvfp4.attention_paged_decode(
                    query,
                    key_cache,
                    value_cache_transposed,
                    query_scale,
                    key_scale,
                    value_scale,
                    block_table,
                    valid_lengths,
                    softmax_scale=softmax_scale,
                    output=output,
                    workspace=workspace,
                )

                logical_key = torch.stack(
                    [
                        key_codes[physical_ids.long()]
                        .permute(1, 0, 2, 3)
                        .reshape(kv_heads, max_kv_length, head_dim)
                        for physical_ids in block_table
                    ]
                )
                logical_value = torch.stack(
                    [
                        value_codes[physical_ids.long()]
                        .permute(1, 2, 0, 3)
                        .reshape(kv_heads, value_dim, max_kv_length)
                        for physical_ids in block_table
                    ]
                )
                query_reference = decode[query_codes.long()] * 0.125
                key_reference = decode[logical_key.long()] * 0.125
                value_reference = decode[logical_value.long()] * 0.5
                kv_head_index = torch.arange(query_heads, device="cuda") // (
                    query_heads // kv_heads
                )
                expanded_key = key_reference[:, kv_head_index]
                expanded_value = value_reference[:, kv_head_index]
                scores = (
                    torch.einsum(
                        "bhd,bhnd->bhn",
                        query_reference,
                        expanded_key,
                    )
                    * softmax_scale
                )
                invalid = torch.arange(max_kv_length, device="cuda").view(
                    1, 1, -1
                ) >= valid_lengths.view(batch, 1, 1)
                scores = scores.masked_fill(invalid, -torch.inf)
                expected = torch.einsum(
                    "bhn,bhdn->bhd",
                    scores.softmax(dim=-1),
                    expanded_value,
                ).to(torch.float16)
                dense_key_scale = torch.full(
                    (
                        batch
                        * kv_heads
                        * sm120_nvfp4.scale_b_elements(
                            1, max_kv_length, head_dim
                        ),
                    ),
                    0x20,
                    dtype=torch.uint8,
                    device="cuda",
                )
                dense_value_scale = torch.full(
                    (
                        batch
                        * kv_heads
                        * sm120_nvfp4.scale_b_elements(
                            1, value_dim, max_kv_length
                        ),
                    ),
                    0x30,
                    dtype=torch.uint8,
                    device="cuda",
                )
                dense_result = sm120_nvfp4.attention_decode(
                    query,
                    self.pack_codes(logical_key),
                    self.pack_codes(logical_value),
                    query_scale,
                    dense_key_scale,
                    dense_value_scale,
                    kv_lengths=valid_lengths,
                    softmax_scale=softmax_scale,
                )
                torch.cuda.synchronize()

                self.assertEqual(result.data_ptr(), output.data_ptr())
                torch.testing.assert_close(
                    result, dense_result, rtol=0, atol=0.002
                )
                torch.testing.assert_close(
                    result, expected, rtol=0.25, atol=0.05
                )


if __name__ == "__main__":
    unittest.main()
