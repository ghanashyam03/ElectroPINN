"""Train supervised temporal transformer baseline."""

from __future__ import annotations

import os
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data.datamodule import BatteryDataModule
from data.preprocess import preprocess_pipeline
from training.callbacks import (
    CheckpointIntegrityCallback,
    GPUMemoryLogger,
    GradientNormLogger,
    NaNGuardCallback,
)
from training.loss_history import LossHistoryCallback
from training.lightning_module import BaselineLitModule
from utils.hydra_config import load_config, project_root
from utils.io import ensure_dir
from utils.logging_utils import get_logger, setup_logging
from utils.seed import set_seed

logger = get_logger(__name__)


def _ensure_processed(cfg: DictConfig) -> None:
    processed = Path(cfg.paths.processed_data)
    if (processed / "train.parquet").exists():
        return
    raw = Path(cfg.paths.raw_data) / "simulations.parquet"
    if not raw.exists():
        raise FileNotFoundError(
            f"Missing raw data at {raw}. Run `make generate-data` first."
        )
    preprocess_pipeline(
        raw_path=raw,
        processed_dir=processed,
        scalers_dir=Path(cfg.paths.scalers),
        train_ratio=float(cfg.train.split.train_ratio),
        val_ratio=float(cfg.train.split.val_ratio),
        test_ratio=float(cfg.train.split.test_ratio),
        train_profiles=list(cfg.train.train_profiles),
        test_profiles=list(cfg.train.test_profiles),
        seed=int(cfg.project.seed),
    )


def _transformer_hparams(cfg: DictConfig) -> dict[str, int | float]:
    t = cfg.model.transformer
    return {
        "input_dim": int(t.input_dim),
        "output_dim": int(t.output_dim),
        "hidden_dim": int(t.hidden_dim),
        "num_layers": int(t.num_layers),
        "num_heads": int(t.num_heads),
        "dropout": float(t.dropout),
    }


def main(cfg: DictConfig) -> None:
    setup_logging()
    os.chdir(project_root())

    fast = os.environ.get("BATTERY_PINN_FAST_TEST", "0") == "1"
    if fast:
        cfg = OmegaConf.merge(cfg, {"train": {"max_epochs": 2, "batch_size": 8}})

    set_seed(int(cfg.project.seed), deterministic=bool(cfg.project.deterministic))
    _ensure_processed(cfg)

    dm = BatteryDataModule(
        processed_dir=cfg.paths.processed_data,
        batch_size=int(cfg.train.batch_size),
        num_workers=int(cfg.train.num_workers),
    )

    lit = BaselineLitModule(
        **_transformer_hparams(cfg),
        learning_rate=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
        max_epochs=int(cfg.train.max_epochs),
    )

    ckpt_dir = ensure_dir(Path(cfg.paths.checkpoints) / "baseline")
    tb_dir = ensure_dir(Path(cfg.paths.tensorboard) / "baseline")

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="baseline-{epoch:02d}-{val_loss:.4f}",
            monitor=str(cfg.train.checkpoint.monitor),
            mode=str(cfg.train.checkpoint.mode),
            save_top_k=int(cfg.train.checkpoint.save_top_k),
        ),
        EarlyStopping(monitor="val_loss", patience=int(cfg.train.early_stopping_patience), mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
        GPUMemoryLogger(),
        GradientNormLogger(),
        NaNGuardCallback(),
        CheckpointIntegrityCallback(),
        LossHistoryCallback(Path(cfg.paths.outputs) / "baseline_loss_history.json"),
    ]

    trainer = pl.Trainer(
        max_epochs=int(cfg.train.max_epochs),
        accelerator=str(cfg.train.accelerator),
        devices=int(cfg.train.devices),
        precision=str(cfg.train.precision),
        gradient_clip_val=float(cfg.train.gradient_clip_val),
        logger=TensorBoardLogger(save_dir=str(tb_dir), name="baseline"),
        callbacks=callbacks,
        log_every_n_steps=int(cfg.train.log_every_n_steps),
        deterministic=bool(cfg.project.deterministic),
    )

    logger.info("Training transformer baseline | params=%s", lit.model.count_parameters())
    trainer.fit(lit, datamodule=dm)
    trainer.test(lit, datamodule=dm)


if __name__ == "__main__":
    main(load_config())
