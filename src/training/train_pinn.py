"""Train physics-guided temporal transformer."""

from __future__ import annotations

import os
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data.datamodule import BatteryDataModule
from models.physics_loss import PhysicsLossConfig
from training.callbacks import (
    CheckpointIntegrityCallback,
    GPUMemoryLogger,
    GradientNormLogger,
    NaNGuardCallback,
)
from training.loss_history import LossHistoryCallback
from training.lightning_module import PINNLitModule
from training.train_baseline import _ensure_processed, _transformer_hparams
from utils.hydra_config import load_config, project_root
from utils.io import ensure_dir
from utils.logging_utils import get_logger, setup_logging
from utils.seed import set_seed

logger = get_logger(__name__)


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

    phycfg = cfg.model.physics
    tcfg = cfg.model.transformer
    physics = PhysicsLossConfig(
        lambda_data=float(phycfg.lambda_data),
        lambda_coulomb=float(phycfg.lambda_coulomb),
        lambda_differential=float(phycfg.lambda_differential),
        lambda_monotonicity=float(phycfg.lambda_monotonicity),
        lambda_voltage_smooth=float(phycfg.lambda_voltage_smooth),
        lambda_voltage_current=float(phycfg.lambda_voltage_current),
        lambda_thermal=float(phycfg.lambda_thermal),
        nominal_capacity_ah=float(tcfg.nominal_capacity_ah),
        use_coulomb=bool(phycfg.use_coulomb),
        use_differential=bool(phycfg.use_differential),
        use_monotonicity=bool(phycfg.use_monotonicity),
        use_voltage_smooth=bool(phycfg.use_voltage_smooth),
        use_voltage_current=bool(phycfg.use_voltage_current),
        use_thermal=bool(phycfg.use_thermal),
    )

    lit = PINNLitModule(
        **_transformer_hparams(cfg),
        learning_rate=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
        max_epochs=int(cfg.train.max_epochs),
        physics_config=physics,
    )

    ckpt_dir = ensure_dir(Path(cfg.paths.checkpoints) / "pinn")
    tb_dir = ensure_dir(Path(cfg.paths.tensorboard) / "pinn")

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="pinn-{epoch:02d}-{val_loss:.4f}",
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
        LossHistoryCallback(Path(cfg.paths.outputs) / "pinn_loss_history.json"),
    ]

    trainer = pl.Trainer(
        max_epochs=int(cfg.train.max_epochs),
        accelerator=str(cfg.train.accelerator),
        devices=int(cfg.train.devices),
        precision=str(cfg.train.precision),
        gradient_clip_val=float(cfg.train.gradient_clip_val),
        logger=TensorBoardLogger(save_dir=str(tb_dir), name="pinn"),
        callbacks=callbacks,
        log_every_n_steps=int(cfg.train.log_every_n_steps),
        deterministic=bool(cfg.project.deterministic),
    )

    logger.info("Training physics-guided transformer | params=%s", lit.model.count_parameters())
    trainer.fit(lit, datamodule=dm)
    trainer.test(lit, datamodule=dm)


if __name__ == "__main__":
    main(load_config())
