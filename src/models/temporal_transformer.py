"""Lightweight causal temporal transformer encoder."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativePositionalEncoding(nn.Module):
    """Relative position bias for causal self-attention (T5-style bucketed distances)."""

    def __init__(self, num_heads: int, max_len: int = 512) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.max_len = max_len
        self.bias = nn.Parameter(torch.zeros(num_heads, max_len, max_len))

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return attention bias (num_heads, seq_len, seq_len)."""
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")
        return self.bias[:, :seq_len, :seq_len].to(device)


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Upper-triangular mask (True = blocked positions)."""
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)


class PreNormTransformerEncoderLayer(nn.Module):
    """PreNorm transformer encoder layer with causal self-attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        ffn_multiplier: int = 4,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_multiplier, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        rel_bias: torch.Tensor,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.norm1(x)
        # Additive mask: causal (-inf upper triangle) + relative position bias
        t = h.shape[1]
        bias = rel_bias.mean(dim=0)
        additive = bias.masked_fill(attn_mask, float("-inf"))
        attn_out, weights = self.attn(
            h,
            h,
            h,
            attn_mask=additive,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, weights


class TemporalTransformerEncoder(nn.Module):
    """Stack of causal PreNorm transformer encoder layers."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.rel_pos = RelativePositionalEncoding(num_heads, max_len=max_len)
        self.layers = nn.ModuleList(
            [
                PreNormTransformerEncoderLayer(hidden_dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | list[torch.Tensor]]:
        """
        Args:
            x: (B, T, H) projected features
        Returns:
            latent: (B, T, H)
            second: causal mask or list of attention weight tensors
        """
        t = x.shape[1]
        device = x.device
        causal = _causal_mask(t, device)
        rel = self.rel_pos(t, device)
        h = x
        attn_weights: list[torch.Tensor] = []
        for layer in self.layers:
            h, w = layer(h, causal, rel, need_weights=return_attention)
            if return_attention and w is not None:
                attn_weights.append(w)
        latent = self.out_norm(h)
        if return_attention:
            return latent, attn_weights
        return latent, causal
