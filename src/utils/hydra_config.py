"""Hydra config loading without @hydra.main (Windows-safe with native solvers)."""

from __future__ import annotations

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig


def project_root() -> Path:
    """Repository root (battery-pinn/)."""
    return Path(__file__).resolve().parents[2]


def load_config(overrides: list[str] | None = None) -> DictConfig:
    """
    Load configs/config.yaml with optional Hydra-style CLI overrides.

    Example overrides: ``simulation.num_simulations=10``
    """
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    config_dir = project_root() / "configs"
    argv_overrides = [a for a in sys.argv[1:] if "=" in a and not a.startswith("-")]
    merged = list(overrides or []) + argv_overrides

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="config", overrides=merged)
