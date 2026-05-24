"""Tests for Monte Carlo dropout uncertainty."""

from __future__ import annotations

import torch

from models.battery_transformer import BatteryStateTransformer
from models.uncertainty import mc_dropout_predict


def test_mc_dropout_shapes() -> None:
    model = BatteryStateTransformer(hidden_dim=32, num_layers=2, num_heads=4, dropout=0.2)
    x = torch.randn(2, 8, 4)
    mean, std = mc_dropout_predict(model, x, n_samples=5)
    assert mean.shape == (2, 8, 3)
    assert std.shape == (2, 8, 3)
    assert (std >= 0).all()
