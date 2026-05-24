"""Tests for physics-guided loss components."""

from __future__ import annotations

import torch

from models.physics_loss import PhysicsLossConfig, PhysicsLossModule


def _batch(b: int = 2, t: int = 16) -> dict[str, torch.Tensor]:
    soc = torch.linspace(1.0, 0.3, t).view(1, t, 1).expand(b, -1, -1)
    return {
        "soc_physical": soc,
        "current": torch.full((b, t), 2.0),
        "dt": torch.full((b, t), 1.0),
        "elapsed_time": torch.linspace(0, t - 1, t).expand(b, -1),
        "voltage": torch.linspace(4.0, 3.5, t).expand(b, -1),
        "temperature": torch.full((b, t), 25.0),
    }


def test_physics_losses_positive() -> None:
    module = PhysicsLossModule(PhysicsLossConfig())
    b = _batch()
    pred = torch.zeros(2, 16, 3)
    pred[..., 0:1] = b["soc_physical"]
    pred[..., 1:2] = b["voltage"].unsqueeze(-1)
    pred[..., 2:3] = b["temperature"].unsqueeze(-1)
    losses = module(pred, b)
    assert losses["loss"].item() >= 0
    for key in ("data", "coulomb", "differential", "monotonicity", "voltage_smooth", "thermal"):
        assert losses[key].item() >= 0


def test_differential_gradient_flow() -> None:
    module = PhysicsLossModule(PhysicsLossConfig())
    b = _batch()
    pred = b["soc_physical"].clone().requires_grad_(True)
    full = torch.cat(
        [pred, b["voltage"].unsqueeze(-1), b["temperature"].unsqueeze(-1)], dim=-1
    )
    loss = module.differential_coulomb_loss(pred, b["elapsed_time"], b["current"])
    loss.backward()
    assert pred.grad is not None
