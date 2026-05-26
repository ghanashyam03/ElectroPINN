"""Compare baseline and physics-guided models with rigorous ID/OOD metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.datamodule import BatterySequenceDataset
from evaluation.ood import collect_latents, fit_mahalanobis, mahalanobis_scores
from evaluation.plots import (
    plot_ablation_results,
    plot_attention_heatmap,
    plot_error_distribution,
    plot_generalization,
    plot_ood_histogram,
    plot_predicted_vs_actual,
    plot_uncertainty_calibration,
    plot_uncertainty_distribution,
    plot_voltage_trajectory,
)
from models.uncertainty import mc_dropout_predict
from training.lightning_module import BaselineLitModule, PINNLitModule
from training.metrics import generalization_score, numpy_metrics
from utils.device import get_device
from utils.io import load_json, save_json


def _latest_checkpoint(directory: Path) -> Path:
    ckpts = sorted(directory.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {directory}")
    return ckpts[-1]


@torch.inference_mode()
def collect_soc_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect SoC predictions and targets."""
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    times: list[np.ndarray] = []
    voltages: list[np.ndarray] = []

    model.eval()
    for batch in dataloader:
        x = batch["features"].to(device)
        out = model(x)
        preds.append(out[..., 0:1].cpu().numpy())
        targets.append(batch["soc_physical"].numpy())
        times.append(batch["dt"].cpu().numpy())
        voltages.append(batch["voltage"].cpu().numpy())

    return (
        np.concatenate(preds, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(times, axis=0),
        np.concatenate(voltages, axis=0),
    )


def _rmse_from_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    pred, tgt, _, _ = collect_soc_predictions(model, loader, device)
    return float(numpy_metrics(pred, tgt)["rmse"])


def compare(
    processed_dir: Path,
    baseline_ckpt_dir: Path,
    pinn_ckpt_dir: Path,
    output_dir: Path,
    batch_size: int = 32,
    mc_samples: int = 20,
) -> dict[str, object]:
    """Run comparison with ID/OOD generalization, uncertainty, and OOD detection."""
    device = get_device()
    output_dir.mkdir(parents=True, exist_ok=True)

    val_ds = BatterySequenceDataset(processed_dir / "val.parquet")
    test_ds = BatterySequenceDataset(processed_dir / "test.parquet")
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    baseline = BaselineLitModule.load_from_checkpoint(
        str(_latest_checkpoint(baseline_ckpt_dir)), map_location=device
    )
    pinn = PINNLitModule.load_from_checkpoint(
        str(_latest_checkpoint(pinn_ckpt_dir)), map_location=device
    )
    baseline.eval().to(device)
    pinn.eval().to(device)

    # ID = val (train profiles), OOD = test (unseen profiles)
    b_id = _rmse_from_loader(baseline, val_loader, device)
    b_ood = _rmse_from_loader(baseline, test_loader, device)
    p_id = _rmse_from_loader(pinn, val_loader, device)
    p_ood = _rmse_from_loader(pinn, test_loader, device)

    b_pred, b_tgt, b_time, b_volt = collect_soc_predictions(baseline, test_loader, device)
    p_pred, p_tgt, _, _ = collect_soc_predictions(pinn, test_loader, device)

    b_metrics = numpy_metrics(b_pred, b_tgt)
    p_metrics = numpy_metrics(p_pred, p_tgt)

    plot_predicted_vs_actual(p_pred, p_tgt, output_dir / "predicted_vs_actual_soc")
    plot_voltage_trajectory(
        b_time[0].ravel(),
        b_volt[0].ravel(),
        None,
        output_dir / "voltage_trajectory",
    )
    plot_error_distribution(p_pred - p_tgt, output_dir / "error_distribution")
    plot_generalization(b_ood, p_ood, output_dir / "unseen_profile_generalization")

    # Uncertainty (physics-guided model)
    batch = next(iter(test_loader))
    x = batch["features"].to(device)
    mean, std = mc_dropout_predict(pinn.model, x, n_samples=mc_samples)
    soc_std = std[..., 0].cpu().numpy()
    soc_err = np.abs(mean[..., 0].cpu().numpy() - batch["soc_physical"][..., 0].numpy())
    plot_uncertainty_calibration(soc_err.ravel(), soc_std.ravel(), output_dir / "uncertainty_calibration")
    plot_uncertainty_distribution(soc_std.ravel(), output_dir / "uncertainty_distribution")

    # OOD detection
    id_lat = collect_latents(pinn, val_loader, device)
    ood_lat = collect_latents(pinn, test_loader, device)
    mahalanobis_params = fit_mahalanobis(id_lat)
    id_scores = mahalanobis_scores(id_lat, mahalanobis_params)
    ood_scores = mahalanobis_scores(ood_lat, mahalanobis_params)
    plot_ood_histogram(id_scores, ood_scores, output_dir / "ood_detection_histogram")

    # Attention visualization (last layer)
    _, attn_list = pinn.model.encode(x[:1], return_attention=True)
    if attn_list:
        plot_attention_heatmap(
            attn_list[-1][0].detach().cpu().numpy(),
            output_dir / "attention_visualization",
        )
    # Ablation table if present
    ablation_path = output_dir.parent / "ablation" / "ablation_metrics.json"
    if ablation_path.exists():
        ablation = load_json(ablation_path)
        plot_ablation_results(ablation, output_dir / "ablation_results")

    results: dict[str, object] = {
        "baseline": {
            **b_metrics,
            "id_rmse": b_id,
            "ood_rmse": b_ood,
            "generalization_score": generalization_score(b_id, b_ood),
        },
        "pinn": {
            **p_metrics,
            "id_rmse": p_id,
            "ood_rmse": p_ood,
            "generalization_score": generalization_score(p_id, p_ood),
        },
        "ood_detection": {
            "id_mean_mahalanobis": float(id_scores.mean()),
            "ood_mean_mahalanobis": float(ood_scores.mean()),
        },
    }
    save_json(output_dir / "comparison_results.json", results)
    return results
