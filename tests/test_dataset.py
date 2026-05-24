"""Tests for dataset preprocessing and dataloaders."""

from __future__ import annotations

from pathlib import Path

import torch

from data.datamodule import BatteryDataModule, BatterySequenceDataset
from data.preprocess import preprocess_pipeline


def test_preprocess_and_datamodule(synthetic_parquet: Path, tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    scalers = tmp_path / "scalers"
    preprocess_pipeline(
        raw_path=synthetic_parquet,
        processed_dir=processed,
        scalers_dir=scalers,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        train_profiles=["constant", "pulse"],
        test_profiles=["random", "wltp"],
        seed=42,
    )
    assert (processed / "train.parquet").exists()
    dm = BatteryDataModule(processed, batch_size=4, num_workers=0, sequence_length=16)
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    assert batch["features"].shape[-1] == 4
    assert batch["soc"].dim() >= 2
    assert "elapsed_time" in batch


def test_dataset_shapes(synthetic_parquet: Path, tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    scalers = tmp_path / "scalers"
    preprocess_pipeline(
        raw_path=synthetic_parquet,
        processed_dir=processed,
        scalers_dir=scalers,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        train_profiles=None,
        test_profiles=None,
        seed=1,
    )
    ds = BatterySequenceDataset(processed / "train.parquet", sequence_length=10)
    sample = ds[0]
    assert sample["features"].shape == (10, 4)
    assert sample["dt"].shape == (10,)
