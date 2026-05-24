"""Mahalanobis OOD detection in transformer latent space."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.inference_mode()
def collect_latents(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Flatten latent vectors from encoder."""
    model.eval()
    chunks: list[np.ndarray] = []
    for batch in dataloader:
        x = batch["features"].to(device)
        if hasattr(model, "model"):
            latent = model.model.encode(x)
        else:
            latent = model.encode(x)
        chunks.append(latent.mean(dim=1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def fit_mahalanobis(id_latents: np.ndarray, eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """Return mean and precision matrix for ID distribution."""
    mean = id_latents.mean(axis=0)
    cov = np.cov(id_latents, rowvar=False) + eps * np.eye(id_latents.shape[1])
    precision = np.linalg.inv(cov)
    return mean, precision


def mahalanobis_scores(latents: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """Compute squared Mahalanobis distance per sample."""
    diff = latents - mean
    return np.einsum("ni,ij,nj->n", diff, precision, diff)
