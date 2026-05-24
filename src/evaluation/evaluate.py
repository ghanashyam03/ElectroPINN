"""Evaluate trained models and generate mandatory plots."""

from __future__ import annotations

import os
from pathlib import Path

from omegaconf import DictConfig

from evaluation.compare_models import compare
from evaluation.plots import plot_physics_loss_curve, plot_training_curve
from training.train_baseline import _ensure_processed
from utils.hydra_config import load_config, project_root
from utils.io import ensure_dir, load_json
from utils.logging_utils import get_logger, setup_logging
from utils.seed import set_seed

logger = get_logger(__name__)


def main(cfg: DictConfig) -> None:
    setup_logging()
    os.chdir(project_root())
    set_seed(int(cfg.project.seed), deterministic=bool(cfg.project.deterministic))
    _ensure_processed(cfg)

    out_dir = ensure_dir(Path(cfg.paths.outputs) / "evaluation")
    results = compare(
        processed_dir=Path(cfg.paths.processed_data),
        baseline_ckpt_dir=Path(cfg.paths.checkpoints) / "baseline",
        pinn_ckpt_dir=Path(cfg.paths.checkpoints) / "pinn",
        output_dir=out_dir,
        batch_size=int(cfg.train.batch_size),
        mc_samples=int(cfg.model.uncertainty.mc_samples),
    )
    _plot_training_histories(Path(cfg.paths.outputs), out_dir)
    logger.info("Evaluation complete: %s", results)


def _plot_training_histories(outputs_dir: Path, plot_dir: Path) -> None:
    """Generate training_curve.png and physics_loss_curve.png from saved histories."""
    baseline_path = outputs_dir / "baseline_loss_history.json"
    pinn_path = outputs_dir / "pinn_loss_history.json"
    if baseline_path.exists():
        hist = load_json(baseline_path)
        if hist.get("train_loss") and hist.get("val_loss"):
            plot_training_curve(
                hist["train_loss"],
                hist["val_loss"],
                plot_dir / "training_curve",
            )
    if pinn_path.exists():
        hist = load_json(pinn_path)
        physics = {
            k.replace("train_", ""): v
            for k, v in hist.items()
            if k.startswith("train_") and k not in ("train_loss",) and v
        }
        if physics:
            plot_physics_loss_curve(physics, plot_dir / "physics_loss_curve")


if __name__ == "__main__":
    main(load_config())
