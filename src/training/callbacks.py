"""Custom PyTorch Lightning callbacks."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

from utils.device import get_gpu_memory_mb


class GPUMemoryLogger(Callback):
    """Log GPU memory usage to Lightning logger."""

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        mem = get_gpu_memory_mb()
        if mem is not None and trainer.logger is not None:
            pl_module.log("gpu_memory_mb", mem, on_step=True, on_epoch=False, prog_bar=False)


class GradientNormLogger(Callback):
    """Log total gradient norm each training step."""

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        norms = []
        for p in pl_module.parameters():
            if p.grad is not None:
                norms.append(p.grad.data.norm(2).item())
        if norms:
            total = sum(n**2 for n in norms) ** 0.5
            pl_module.log("grad_norm", total, on_step=True, on_epoch=False)


class NaNGuardCallback(Callback):
    """Detect NaN/Inf losses and stop training early."""

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        if outputs is None:
            return
        loss: torch.Tensor | None
        if isinstance(outputs, dict):
            loss = outputs.get("loss")  # type: ignore[assignment]
        elif isinstance(outputs, torch.Tensor):
            loss = outputs
        else:
            return
        if loss is not None and not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss detected at step {trainer.global_step}")


class CheckpointIntegrityCallback(Callback):
    """Validate checkpoint file exists after save."""

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, object],
    ) -> None:
        if not checkpoint:
            raise RuntimeError("Empty checkpoint dict")
