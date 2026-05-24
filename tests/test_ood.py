"""Tests for Mahalanobis OOD detection."""

from __future__ import annotations

import numpy as np

from evaluation.ood import fit_mahalanobis, mahalanobis_scores


def test_mahalanobis_separation() -> None:
    rng = np.random.default_rng(0)
    id_lat = rng.normal(0, 1, size=(100, 8))
    ood_lat = rng.normal(3, 1, size=(50, 8))
    mean, precision = fit_mahalanobis(id_lat)
    id_s = mahalanobis_scores(id_lat, mean, precision)
    ood_s = mahalanobis_scores(ood_lat, mean, precision)
    assert ood_s.mean() > id_s.mean()
