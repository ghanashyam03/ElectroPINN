"""Evaluation metrics for battery state estimation."""

from __future__ import annotations

import numpy as np
import torch


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def max_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.abs(pred - target))


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-8)


def numpy_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute metrics in NumPy for reporting."""
    err = pred - target
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "max_error": float(np.max(np.abs(err))),
        "r2": float(1.0 - ss_res / (ss_tot + 1e-8)),
    }


def generalization_score(id_rmse: float, ood_rmse: float) -> float:
    """
    Generalization score = OOD_RMSE / ID_RMSE.

    ID_RMSE: validation on train-profile distributions.
    OOD_RMSE: evaluation on unseen-profile distributions.
    Lower is better (closer to 1.0 means similar ID/OOD error).
    """
    return ood_rmse / (id_rmse + 1e-8)
