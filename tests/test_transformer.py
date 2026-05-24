"""Transformer-specific tests."""

from __future__ import annotations

import torch

from models.battery_transformer import BatteryStateTransformer
from models.uncertainty import mc_dropout_predict
from utils.seed import set_seed


def test_deterministic_inference() -> None:
    set_seed(0)
    model = BatteryStateTransformer(hidden_dim=32, num_layers=2, num_heads=4, dropout=0.0)
    model.eval()
    x = torch.randn(1, 12, 4)
    with torch.inference_mode():
        a = model(x)
        b = model(x)
    assert torch.allclose(a, b)


def test_fair_output_heads() -> None:
    model = BatteryStateTransformer()
    y = model(torch.randn(1, 10, 4))
    assert y.shape[-1] == 3
