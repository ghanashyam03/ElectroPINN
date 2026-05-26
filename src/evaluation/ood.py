"""Mahalanobis OOD detection in transformer latent space."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf


class MahalanobisParams(dict):
    """Custom dict that can be unpacked as (mean, precision) for backward compatibility."""

    def __iter__(self):
        yield self["mean"]
        yield self["precision"]


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


def fit_mahalanobis(id_latents: np.ndarray, eps: float = 1e-5) -> MahalanobisParams:
    """Return fit parameters dictionary for robust OOD detection on ID distribution."""
    # STEP 1: Whitening / Center (zero-mean the latents)
    mean = id_latents.mean(axis=0)
    centered_latents = id_latents - mean

    n_samples, latent_dim = id_latents.shape

    # STEP 2: Dimensionality reduction via PCA before covariance estimation
    pca = None
    if n_samples < 5 * latent_dim:
        n_components = max(8, min(latent_dim, n_samples // 5))
        # Safeguard if total samples or dimensions is less than the floor
        n_components = min(n_components, n_samples, latent_dim)
        pca = PCA(n_components=n_components)
        processed_latents = pca.fit_transform(centered_latents)
    else:
        processed_latents = centered_latents

    # STEP 3: Regularized covariance estimation via LedoitWolf
    lw = LedoitWolf()
    lw.fit(processed_latents)
    precision = lw.precision_

    # STEP 4: Score normalization on ID set
    raw_scores = np.einsum("ni,ij,ni->n", processed_latents, precision, processed_latents)
    score_mean = float(raw_scores.mean())
    score_std = max(float(raw_scores.std()), 1e-6)

    return MahalanobisParams({
        "mean": mean,
        "pca": pca,
        "precision": precision,
        "score_mean": score_mean,
        "score_std": score_std,
    })


def mahalanobis_scores(
    latents: np.ndarray,
    params_or_mean: dict[str, any] | np.ndarray,
    precision: np.ndarray | None = None,
) -> np.ndarray:
    """Compute normalized (or classic) Mahalanobis distance per sample."""
    if precision is not None or isinstance(params_or_mean, np.ndarray):
        # Backward compatibility for old 3-argument signature: (latents, mean, precision)
        mean = params_or_mean
        diff = latents - mean
        return np.einsum("ni,ij,ni->n", diff, precision, diff)

    # New dict-based interface: (latents, params)
    params = params_or_mean
    mean = params["mean"]
    pca = params["pca"]
    prec = params["precision"]
    score_mean = params["score_mean"]
    score_std = params["score_std"]

    # center
    centered = latents - mean

    # PCA transform (if pca is not None)
    if pca is not None:
        processed = pca.transform(centered)
    else:
        processed = centered

    # Mahalanobis distance
    raw_scores = np.einsum("ni,ij,ni->n", processed, prec, processed)

    # normalize with stored mean/std
    return (raw_scores - score_mean) / score_std
