"""Inference utility tests."""

from __future__ import annotations

import torch

from models.battery_transformer import BatteryStateTransformer
from models.inference import measure_latency, predict_soc


def test_predict_soc_and_latency() -> None:
    model = BatteryStateTransformer(hidden_dim=32, num_layers=2, num_heads=4)
    device = torch.device("cpu")
    x = torch.randn(16, 8, 4)
    pred = predict_soc(model, x, device)
    assert pred.shape == (16, 8, 1)
    ms = measure_latency(model, x, device, warmup=1, repeats=5)
    assert ms > 0
