"""Tests for temporal transformer forward and backward passes."""

from __future__ import annotations

import torch

from models.battery_transformer import BatteryStateTransformer
from models.temporal_transformer import _causal_mask


def test_causal_mask() -> None:
    m = _causal_mask(8, torch.device("cpu"))
    assert m.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            if j > i:
                assert m[i, j]
            else:
                assert not m[i, j]


def test_transformer_forward_backward() -> None:
    model = BatteryStateTransformer(hidden_dim=64, num_layers=2, num_heads=4)
    x = torch.randn(4, 16, 4, requires_grad=True)
    y = model(x)
    assert y.shape == (4, 16, 3)
    assert model.count_parameters() < 2_000_000
    y.mean().backward()
    assert x.grad is not None


def test_attention_shapes() -> None:
    model = BatteryStateTransformer(hidden_dim=32, num_layers=2, num_heads=4)
    x = torch.randn(2, 10, 4)
    latent, attn = model.encode(x, return_attention=True)
    assert latent.shape == (2, 10, 32)
    assert len(attn) == 2
    assert attn[-1].shape[1] == 4  # num_heads
