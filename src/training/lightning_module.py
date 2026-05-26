"""PyTorch Lightning modules for transformer baseline and physics-guided models."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn

from models.battery_transformer import BatteryStateTransformer
from models.physics_loss import PhysicsLossConfig, PhysicsLossModule, PhysicsLossToggles
from training.metrics import mae, max_error, r2_score, rmse
from utils.device import get_gpu_memory_mb


def _supervised_loss(
    pred: torch.Tensor,
    batch: dict[str, torch.Tensor],
    mse: nn.MSELoss,
    huber_soc: nn.HuberLoss,
) -> torch.Tensor:
    """Identical supervision for baseline and physics-guided data term."""
    pred_soc = pred[..., 0:1]
    target_soc = batch["soc_physical"]

    # Component 1 & 2: Huber loss and boundary emphasis weighting
    raw_loss = huber_soc(pred_soc, target_soc)
    weights = 1.0 + 0.5 * torch.abs(target_soc - 0.5)
    weighted_loss = (raw_loss * weights).mean()

    # Component 3: Range consistency penalty
    pred_range = pred_soc.max() - pred_soc.min()
    target_range = target_soc.max() - target_soc.min()
    range_loss = torch.relu(target_range - pred_range)
    total_soc_loss = weighted_loss + 0.2 * range_loss

    loss = total_soc_loss + mse(pred[..., 1:2], batch["voltage"].unsqueeze(-1))
    loss = loss + mse(pred[..., 2:3], batch["temperature"].unsqueeze(-1))
    return loss


def _log_soc_metrics(
    module: pl.LightningModule,
    pred: torch.Tensor,
    batch: dict[str, torch.Tensor],
    stage: str,
) -> None:
    pred_soc = pred[..., 0:1]
    target = batch["soc_physical"]
    module.log(f"{stage}_rmse", rmse(pred_soc, target), on_epoch=True)
    module.log(f"{stage}_mae", mae(pred_soc, target), on_epoch=True)
    module.log(f"{stage}_r2", r2_score(pred_soc, target), on_epoch=True)
    module.log(f"{stage}_max_error", max_error(pred_soc, target), on_epoch=True)


class BaselineLitModule(pl.LightningModule):
    """Supervised temporal transformer (no physics-guided terms)."""

    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        max_epochs: int = 30,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = BatteryStateTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.loss_fn = nn.MSELoss()
        self.huber_soc = nn.HuberLoss(delta=0.1, reduction="none")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        pred = self(batch["features"])
        loss = _supervised_loss(pred, batch, self.loss_fn, self.huber_soc)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        _log_soc_metrics(self, pred, batch, stage)
        mem = get_gpu_memory_mb()
        if mem is not None:
            self.log(f"{stage}_gpu_memory_mb", mem, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


class PINNLitModule(pl.LightningModule):
    """Physics-guided temporal transformer."""

    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        max_epochs: int = 30,
        physics_config: PhysicsLossConfig | None = None,
        physics_toggles: PhysicsLossToggles | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["physics_config", "physics_toggles"])
        self.model = BatteryStateTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        cfg = physics_config or PhysicsLossConfig()
        self.physics = PhysicsLossModule(cfg, toggles=physics_toggles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        pred = self(batch["features"])
        losses = self.physics(pred, batch)
        for name, value in losses.items():
            self.log(f"{stage}_{name}", value, prog_bar=(name == "loss"), on_epoch=True)
        _log_soc_metrics(self, pred, batch, stage)
        mem = get_gpu_memory_mb()
        if mem is not None:
            self.log(f"{stage}_gpu_memory_mb", mem, on_epoch=True)
        return losses["loss"]

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
