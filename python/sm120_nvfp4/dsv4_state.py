"""Per-request CSA cache and bounded overlapping compressor state."""
from __future__ import annotations

from dataclasses import dataclass, fields
import math

import torch


@dataclass(frozen=True)
class DSV4CSAConfig:
    hidden_size: int = 4096
    num_attention_heads: int = 64
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    sliding_window: int = 128
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    compress_ratio: int = 4
    rms_norm_eps: float = 1.e-6
    compress_rope_theta: float = 160000.
    original_max_position_embeddings: int = 65536
    rope_factor: float = 16.
    beta_fast: float = 32.
    beta_slow: float = 1.
    max_position_embeddings: int = 1048576
    hc_mult: int = 4
    hc_eps: float = 1.e-6
    hc_sinkhorn_iters: int = 20

    def __post_init__(self):
        fixed = (self.num_attention_heads, self.head_dim, self.qk_rope_head_dim,
                 self.sliding_window, self.index_n_heads, self.index_head_dim,
                 self.index_topk, self.compress_ratio)
        if fixed != (64, 512, 64, 128, 64, 128, 512, 4):
            raise ValueError("this block supports DS-V4-Flash CSA: H=64,D=512,RoPE=64,SWA=128,index=64x128,TopK=512,ratio=4")
        if any(type(v) is not int or v <= 0 for v in
               (self.hidden_size, self.q_lora_rank, self.o_lora_rank, self.o_groups,
                self.max_position_embeddings, self.hc_mult, self.hc_sinkhorn_iters)):
            raise ValueError("dimensions, capacities and iteration counts must be positive integers")
        if self.num_attention_heads % self.o_groups or self.max_position_embeddings > 1048576:
            raise ValueError("output groups must divide heads; context limit is 1048576")
        numeric = (self.rms_norm_eps, self.hc_eps, self.compress_rope_theta,
                   self.rope_factor, self.beta_fast, self.beta_slow)
        if not all(math.isfinite(v) for v in numeric):
            raise ValueError("norm/RoPE parameters must be finite")
        if not (self.rms_norm_eps > 0 and self.hc_eps > 0 and self.compress_rope_theta > 1 and
                self.rope_factor >= 1 and self.beta_fast > self.beta_slow > 0 and
                self.original_max_position_embeddings >= 0):
            raise ValueError("invalid norm/RoPE parameters")

    @classmethod
    def from_hf_config(cls, config, layer_id=2):
        if config.get("model_type") != "deepseek_v4" or config.get("num_key_value_heads") != 1:
            raise ValueError("expected a DeepSeek-V4 shared-KV checkpoint")
        ratios = config.get("compress_ratios", [])
        if not 0 <= layer_id < len(ratios) or ratios[layer_id] != 4:
            raise ValueError("select a CSA layer with compress_ratios[layer_id] == 4")
        yarn = config.get("rope_scaling", {})
        if yarn.get("type", yarn.get("rope_type")) != "yarn":
            raise ValueError("expected the official YaRN configuration")
        names = {f.name for f in fields(cls)}
        values = {k: v for k, v in config.items() if k in names}
        values.update(compress_ratio=4,
                      original_max_position_embeddings=yarn["original_max_position_embeddings"],
                      rope_factor=yarn["factor"], beta_fast=yarn["beta_fast"], beta_slow=yarn["beta_slow"])
        return cls(**values)


class CompressionState:
    def __init__(self, dim, device):
        self.previous_values = torch.zeros((4, dim), dtype=torch.float32, device=device)
        self.previous_scores = torch.full_like(self.previous_values, -torch.inf)
        self.pending_values = torch.empty((0, 2 * dim), dtype=torch.float32, device=device)
        self.pending_scores = torch.empty_like(self.pending_values)

    def fork(self):
        result = object.__new__(type(self))
        for key, value in vars(self).items():
            setattr(result, key, value.clone())
        return result


class DSV4CSAState:
    """One request on one CUDA stream. Fork before branching a prefix.

    No tensor is shared by forked states. The last complete block contributes
    its first projection half to the next compression window; at most three
    uncommitted tokens retain both halves. No historical hidden states persist.
    """
    def __init__(self, config, max_seq_len, device, owner):
        if type(max_seq_len) is not int or not 1 <= max_seq_len <= config.max_position_embeddings:
            raise ValueError("max_seq_len is outside the configured context capacity")
        self.config, self.max_seq_len, self.device = config, max_seq_len, torch.device(device)
        self._owner, self._failed = owner, False
        self._stream = torch.cuda.current_stream(self.device).cuda_stream
        self.length = 0
        self.swa_cache = torch.zeros((2, 64, 384), device=device, dtype=torch.uint8)
        capacity = max_seq_len // 4
        self.compressed_cache = torch.zeros(((capacity + 63) // 64, 64, 384), device=device, dtype=torch.uint8)
        self.index_payload = torch.zeros((capacity, 64), device=device, dtype=torch.uint8)
        self.index_scales = torch.ones((capacity, 4), device=device, dtype=torch.uint8)
        self.attention_compressor = CompressionState(512, device)
        self.index_compressor = CompressionState(128, device)

    @property
    def compressed_length(self):
        return self.length // 4

    def _check_stream(self):
        if torch.cuda.current_stream(self.device).cuda_stream != self._stream:
            raise ValueError("CSA state must be used on its creating CUDA stream")

    def fork(self):
        self._check_stream()
        if self._failed:
            raise ValueError("cannot fork a failed state; reset it first")
        result = object.__new__(type(self))
        for key, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                value = value.clone()
            elif isinstance(value, CompressionState):
                value = value.fork()
            setattr(result, key, value)
        return result

    def reset(self):
        self._check_stream()
        self.swa_cache.zero_()
        self.compressed_cache.zero_()
        self.index_payload.zero_()
        self.index_scales.fill_(1)
        self.attention_compressor = CompressionState(512, self.device)
        self.index_compressor = CompressionState(128, self.device)
        self.length, self._failed = 0, False

    def tensors(self):
        """Named cache/state tensors for server diagnostics; no device reads."""
        result = {name: value for name, value in vars(self).items() if isinstance(value, torch.Tensor)}
        for name in ("attention_compressor", "index_compressor"):
            result.update({f"{name}.{key}": value for key, value in vars(getattr(self, name)).items()})
        return result
