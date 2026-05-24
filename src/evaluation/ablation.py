"""Automated ablation study for physics-guided loss components."""

from __future__ import annotations

import os
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from data.datamodule import BatteryDataModule, BatterySequenceDataset
from evaluation.compare_models import _rmse_from_loader
from evaluation.plots import plot_ablation_results
from models.physics_loss import PhysicsLossConfig, PhysicsLossToggles
from training.lightning_module import BaselineLitModule, PINNLitModule
from training.metrics import generalization_score
from training.train_baseline import _ensure_processed, _transformer_hparams
from utils.hydra_config import load_config, project_root
from utils.device import get_device
from utils.io import ensure_dir, save_json
from utils.logging_utils import get_logger, setup_logging
from utils.seed import set_seed

logger = get_logger(__name__)


def _run_variant(
    cfg: DictConfig,
    name: str,
    use_physics: bool,
    toggles: PhysicsLossToggles,
    dm: BatteryDataModule,
    max_epochs: int,
) -> dict[str, float]:
    device = get_device()
    hparams = _transformer_hparams(cfg)

    if use_physics:
        phycfg = cfg.model.physics
        physics = PhysicsLossConfig(
            lambda_data=float(phycfg.lambda_data),
            lambda_coulomb=float(phycfg.lambda_coulomb),
            lambda_differential=float(phycfg.lambda_differential),
            lambda_monotonicity=float(phycfg.lambda_monotonicity),
            lambda_voltage_smooth=float(phycfg.lambda_voltage_smooth),
            lambda_voltage_current=float(phycfg.lambda_voltage_current),
            lambda_thermal=float(phycfg.lambda_thermal),
            nominal_capacity_ah=float(cfg.model.transformer.nominal_capacity_ah),
        )
        lit: pl.LightningModule = PINNLitModule(
            **hparams,
            max_epochs=max_epochs,
            physics_config=physics,
            physics_toggles=toggles,
        )
    else:
        lit = BaselineLitModule(**hparams, max_epochs=max_epochs)

    ckpt_dir = ensure_dir(Path(cfg.paths.checkpoints) / "ablation" / name)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=str(cfg.train.accelerator),
        devices=int(cfg.train.devices),
        precision=str(cfg.train.precision),
        gradient_clip_val=float(cfg.train.gradient_clip_val),
        logger=False,
        enable_progress_bar=False,
        callbacks=[
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                filename="best",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            ),
            EarlyStopping(monitor="val_loss", patience=5, mode="min"),
        ],
    )
    trainer.fit(lit, datamodule=dm)

    val_loader = DataLoader(
        BatterySequenceDataset(Path(cfg.paths.processed_data) / "val.parquet"),
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
    )
    test_loader = DataLoader(
        BatterySequenceDataset(Path(cfg.paths.processed_data) / "test.parquet"),
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
    )
    id_rmse = _rmse_from_loader(lit, val_loader, device)
    ood_rmse = _rmse_from_loader(lit, test_loader, device)
    return {
        "id_rmse": id_rmse,
        "ood_rmse": ood_rmse,
        "generalization_score": generalization_score(id_rmse, ood_rmse),
    }


def main(cfg: DictConfig) -> None:
    setup_logging()
    os.chdir(project_root())
    fast = os.environ.get("BATTERY_PINN_FAST_TEST", "0") == "1"
    max_epochs = 2 if fast else int(cfg.model.ablation.max_epochs)

    set_seed(int(cfg.project.seed), deterministic=bool(cfg.project.deterministic))
    _ensure_processed(cfg)

    dm = BatteryDataModule(
        processed_dir=cfg.paths.processed_data,
        batch_size=int(cfg.train.batch_size),
        num_workers=int(cfg.train.num_workers),
    )

    results: dict[str, dict[str, float]] = {}
    for variant in cfg.model.ablation.variants:
        name = str(variant.name)
        toggles_dict = OmegaConf.to_container(variant.get("toggles", {}), resolve=True)
        assert isinstance(toggles_dict, dict)
        base = PhysicsLossToggles()
        merged = {**base.__dict__, **{k: bool(v) for k, v in toggles_dict.items()}}
        toggles = PhysicsLossToggles(**merged)
        use_physics = bool(variant.get("physics", False))
        logger.info("Ablation variant: %s", name)
        results[name] = _run_variant(cfg, name, use_physics, toggles, dm, max_epochs)

    out_dir = ensure_dir(Path(cfg.paths.outputs) / "ablation")
    save_json(out_dir / "ablation_metrics.json", results)
    plot_ablation_results(results, out_dir / "ablation_results")
    logger.info("Ablation complete: %s", results)


if __name__ == "__main__":
    main(load_config())
