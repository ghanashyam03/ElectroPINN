"""Ablation pipeline smoke test."""

from __future__ import annotations

from pathlib import Path

import torch

from models.physics_loss import PhysicsLossToggles
from training.lightning_module import BaselineLitModule, PINNLitModule


def test_ablation_modules(synthetic_parquet: Path, tmp_path: Path) -> None:
    from data.datamodule import BatteryDataModule
    from data.preprocess import preprocess_pipeline

    processed = tmp_path / "processed"
    preprocess_pipeline(
        raw_path=synthetic_parquet,
        processed_dir=processed,
        scalers_dir=tmp_path / "scalers",
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        train_profiles=["constant", "pulse"],
        test_profiles=["random", "wltp"],
        seed=0,
    )
    dm = BatteryDataModule(processed, batch_size=2, num_workers=0, sequence_length=10)
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))

    baseline = BaselineLitModule(hidden_dim=32, num_layers=2, num_heads=4)
    pinn = PINNLitModule(
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        physics_toggles=PhysicsLossToggles(coulomb=True, monotonicity=True),
    )
    b_out = baseline(batch["features"])
    p_out = pinn(batch["features"])
    assert b_out.shape == p_out.shape == (2, 10, 3)
