"""Device selection and GPU memory utilities."""

from __future__ import annotations

import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return CUDA device if available and requested, else CPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_gpu_memory_mb() -> float | None:
    """Return allocated GPU memory in MB, or None if CUDA unavailable."""
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.memory_allocated() / (1024**2))
