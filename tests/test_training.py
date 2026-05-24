"""Smoke tests for Lightning training loops."""

from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import torch

from data.datamodule import BatteryDataModule
from data.preprocess import preprocess_pipeline
from models.physics_loss import PhysicsLossConfig
from training.lightning_module import BaselineLitModule, PINNLitModule


def test_training_smoke(synthetic_parquet: Path, tmp_path: Path) -> None:
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
    dm = BatteryDataModule(processed, batch_size=4, num_workers=0, sequence_length=12)
    baseline = BaselineLitModule(hidden_dim=32, num_layers=2, num_heads=4, max_epochs=2, dropout=0.0)
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )
    trainer.fit(baseline, datamodule=dm)

    pinn = PINNLitModule(
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        max_epochs=2,
        dropout=0.0,
        physics_config=PhysicsLossConfig(),
    )
    trainer.fit(pinn, datamodule=dm)


def test_checkpoint_roundtrip(synthetic_parquet: Path, tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    preprocess_pipeline(
        raw_path=synthetic_parquet,
        processed_dir=processed,
        scalers_dir=tmp_path / "scalers",
        train_ratio=0.75,
        val_ratio=0.15,
        test_ratio=0.10,
        train_profiles=None,
        test_profiles=None,
        seed=3,
    )
    dm = BatteryDataModule(processed, batch_size=2, num_workers=0, sequence_length=8)
    ckpt_path = tmp_path / "model.ckpt"
    model = BaselineLitModule(hidden_dim=16, num_layers=1, num_heads=4, dropout=0.0)
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        callbacks=[pl.callbacks.ModelCheckpoint(dirpath=tmp_path, filename="model")],
        logger=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, datamodule=dm)
    loaded = BaselineLitModule.load_from_checkpoint(str(list(tmp_path.glob("*.ckpt"))[0]))
    model.eval()
    loaded.eval()
    batch = next(iter(dm.train_dataloader()))
    with torch.inference_mode():
        out1 = model(batch["features"])
        out2 = loaded(batch["features"])
    assert torch.allclose(out1, out2, atol=1e-4)
