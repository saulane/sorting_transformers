from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


def build_absolute_positional_embedding(max_len: int, d_model: int) -> nn.Embedding:
    return nn.Embedding(max_len, d_model)


@dataclass
class RelativePositionBias:
    n_heads: int
    max_len: int
    shared: bool = False

    def __post_init__(self) -> None:
        num_heads = 1 if self.shared else self.n_heads
        self.weight = nn.Parameter(torch.zeros(num_heads, 2 * self.max_len - 1))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def get_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if seq_len > self.max_len:
            raise ValueError("seq_len exceeds max_len for relative bias")
        positions = torch.arange(seq_len, device=device)
        rel = positions[None, :] - positions[:, None]
        rel = rel.clamp(-(self.max_len - 1), self.max_len - 1) + (self.max_len - 1)
        bias = self.weight[:, rel]
        if self.shared:
            bias = bias.expand(self.n_heads, -1, -1)
        return bias


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: int = 10000) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2)
        device = q.device
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        q_rot = apply_rotary(q, cos, sin)
        k_rot = apply_rotary(k, cos, sin)
        return q_rot, k_rot


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


@dataclass
class AlibiBias:
    n_heads: int

    def get_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        slopes = _alibi_slopes(self.n_heads).to(device)
        position = torch.arange(seq_len, device=device)
        rel = position[None, :] - position[:, None]
        rel = rel.abs().float()
        bias = -rel.unsqueeze(0) * slopes[:, None, None]
        return bias


def _alibi_slopes(n_heads: int) -> torch.Tensor:
    def get_slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-2 ** -(torch.log2(torch.tensor(n)).item() - 3))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if (n_heads & (n_heads - 1)) == 0:
        slopes = get_slopes_power_of_2(n_heads)
    else:
        closest_power = 2 ** (n_heads.bit_length() - 1)
        slopes = get_slopes_power_of_2(closest_power)
        extra = get_slopes_power_of_2(2 * closest_power)[0::2][: n_heads - closest_power]
        slopes.extend(extra)
    return torch.tensor(slopes)


class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        return self.pos_emb(positions)


@dataclass
class PositionalSpec:
    absolute: Optional[PositionalEncoding] = None
    relative_bias: Optional[RelativePositionBias] = None
    rope: Optional[RotaryEmbedding] = None
    alibi: Optional[AlibiBias] = None
