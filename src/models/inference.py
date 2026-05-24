"""Inference utilities for trained models."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from models.baseline_mlp import BaselineMLP
from models.pinn_model import PINNModel


def load_baseline(checkpoint_path: str | Path, device: torch.device | None = None) -> BaselineMLP:
    """Load baseline MLP weights from a Lightning checkpoint."""
    from training.lightning_module import BaselineLitModule

    device = device or torch.device("cpu")
    lit = BaselineLitModule.load_from_checkpoint(str(checkpoint_path), map_location=device)
    lit.model.eval()
    return lit.model.to(device)


def load_model_from_state(
    state_dict: dict[str, torch.Tensor],
    model_type: str,
    model_kwargs: dict[str, object],
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Instantiate model and load weights from state dict."""
    device = device or torch.device("cpu")
    if model_type == "baseline":
        model: torch.nn.Module = BaselineMLP(**model_kwargs)  # type: ignore[arg-type]
    elif model_type == "pinn":
        model = PINNModel(**model_kwargs)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)


@torch.inference_mode()
def predict_soc(
    model: torch.nn.Module,
    features: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Run forward pass and return SoC predictions."""
    model.eval()
    x = features.to(device)
    out = model(x)
    if out.dim() == 3 and out.shape[-1] > 1:
        return out[..., 0:1]
    return out


@torch.inference_mode()
def measure_latency(
    model: torch.nn.Module,
    features: torch.Tensor,
    device: torch.device,
    warmup: int = 5,
    repeats: int = 50,
) -> float:
    """Measure mean inference latency in milliseconds."""
    model.eval()
    x = features.to(device)
    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / repeats
    return elapsed_ms
