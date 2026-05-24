"""Monte Carlo dropout uncertainty estimation."""

from __future__ import annotations

import torch
import torch.nn as nn


def enable_dropout(model: nn.Module) -> None:
    """Activate dropout layers for stochastic forward passes."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_predict(
    model: nn.Module,
    features: torch.Tensor,
    n_samples: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run MC dropout inference.

    Returns:
        mean: (B, T, C)
        std: (B, T, C)
    """
    was_training = model.training
    model.eval()
    enable_dropout(model)
    samples: list[torch.Tensor] = []
    with torch.inference_mode():
        for _ in range(n_samples):
            samples.append(model(features))
    if was_training:
        model.train()
    stacked = torch.stack(samples, dim=0)
    return stacked.mean(dim=0), stacked.std(dim=0)
