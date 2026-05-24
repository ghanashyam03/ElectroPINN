"""Persist training metrics for offline plotting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pytorch_lightning.callbacks import Callback


class LossHistoryCallback(Callback):
    """Save per-epoch losses to JSON for evaluation plots."""

    def __init__(self, output_path: str | Path) -> None:
        super().__init__()
        self.output_path = Path(output_path)
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_data": [],
            "train_coulomb": [],
            "train_monotonicity": [],
            "train_voltage": [],
            "train_thermal": [],
        }

    def on_validation_epoch_end(self, trainer: object, pl_module: object) -> None:
        metrics = trainer.callback_metrics  # type: ignore[attr-defined]
        mapping = {
            "train_loss": "train_loss",
            "val_loss": "val_loss",
            "train_data": "train_data",
            "train_coulomb": "train_coulomb",
            "train_monotonicity": "train_monotonicity",
            "train_voltage": "train_voltage",
            "train_thermal": "train_thermal",
        }
        for key, metric_name in mapping.items():
            if metric_name in metrics:
                self.history[key].append(float(metrics[metric_name]))

    def on_fit_end(self, trainer: object, pl_module: object) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
