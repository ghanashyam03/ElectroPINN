"""Plotly/Matplotlib visualization for battery state estimation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def _save_plotly(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path.with_suffix(".html")))
    try:
        fig.write_image(str(path.with_suffix(".png")), scale=2)
    except Exception:
        fig.write_image(str(path.with_suffix(".png")), engine="kaleido", scale=2)


def plot_predicted_vs_actual(
    pred: np.ndarray,
    actual: np.ndarray,
    output_path: Path,
) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=actual.ravel(),
            y=pred.ravel(),
            mode="markers",
            marker={"size": 4, "opacity": 0.5},
        )
    )
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash"}))
    fig.update_layout(
        title="Predicted vs Actual SoC",
        xaxis_title="Actual SoC",
        yaxis_title="Predicted SoC",
    )
    _save_plotly(fig, output_path)


def plot_voltage_trajectory(
    time: np.ndarray,
    voltage: np.ndarray,
    pred_voltage: np.ndarray | None,
    output_path: Path,
) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=voltage, name="measured"))
    if pred_voltage is not None:
        fig.add_trace(go.Scatter(x=time, y=pred_voltage, name="predicted"))
    fig.update_layout(title="Voltage Trajectory", xaxis_title="Time [s]", yaxis_title="V")
    _save_plotly(fig, output_path)


def plot_training_curve(
    train_loss: list[float],
    val_loss: list[float],
    output_path: Path,
) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_loss, name="train"))
    fig.add_trace(go.Scatter(y=val_loss, name="val"))
    fig.update_layout(title="Training Curve", xaxis_title="Epoch", yaxis_title="Loss")
    _save_plotly(fig, output_path)


def plot_physics_loss_curve(
    loss_history: dict[str, list[float]],
    output_path: Path,
) -> None:
    fig = go.Figure()
    for name, values in loss_history.items():
        fig.add_trace(go.Scatter(y=values, name=name))
    fig.update_layout(title="Physics Loss Components", xaxis_title="Epoch", yaxis_title="Loss")
    _save_plotly(fig, output_path)


def plot_error_distribution(errors: np.ndarray, output_path: Path) -> None:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=errors.ravel(), nbinsx=50))
    fig.update_layout(title="SoC Error Distribution", xaxis_title="Error")
    _save_plotly(fig, output_path)
    plt.figure(figsize=(8, 5))
    plt.hist(errors.ravel(), bins=50, alpha=0.75)
    plt.xlabel("SoC error")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path.with_suffix(".png"), dpi=150)
    plt.close()


def plot_generalization(
    baseline_rmse: float,
    pinn_rmse: float,
    output_path: Path,
) -> None:
    fig = go.Figure(
        data=[
            go.Bar(
                x=["Transformer baseline", "Physics-guided"],
                y=[baseline_rmse, pinn_rmse],
                text=[f"{baseline_rmse:.4f}", f"{pinn_rmse:.4f}"],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(title="OOD RMSE (unseen profiles)", yaxis_title="RMSE")
    _save_plotly(fig, output_path)


def plot_uncertainty_calibration(
    errors: np.ndarray,
    std: np.ndarray,
    output_path: Path,
) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=std.ravel(), y=errors.ravel(), mode="markers", marker={"size": 4}))
    fig.update_layout(title="Uncertainty Calibration", xaxis_title="Std", yaxis_title="|Error|")
    _save_plotly(fig, output_path)
    plt.figure(figsize=(8, 5))
    plt.scatter(std.ravel(), errors.ravel(), s=6, alpha=0.4)
    plt.xlabel("Predictive std")
    plt.ylabel("|SoC error|")
    plt.tight_layout()
    plt.savefig(output_path.with_suffix(".png"), dpi=150)
    plt.close()


def plot_uncertainty_distribution(std: np.ndarray, output_path: Path) -> None:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=std.ravel(), nbinsx=40))
    fig.update_layout(title="Uncertainty Distribution", xaxis_title="Std")
    _save_plotly(fig, output_path)
    plt.figure(figsize=(8, 5))
    plt.hist(std.ravel(), bins=40, alpha=0.8)
    plt.xlabel("Predictive std")
    plt.tight_layout()
    plt.savefig(output_path.with_suffix(".png"), dpi=150)
    plt.close()


def plot_ood_histogram(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    output_path: Path,
) -> None:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=id_scores, name="ID", opacity=0.6))
    fig.add_trace(go.Histogram(x=ood_scores, name="OOD", opacity=0.6))
    fig.update_layout(title="OOD Detection", xaxis_title="Mahalanobis distance")
    _save_plotly(fig, output_path)
    plt.figure(figsize=(8, 5))
    plt.hist(id_scores, bins=40, alpha=0.6, label="ID")
    plt.hist(ood_scores, bins=40, alpha=0.6, label="OOD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path.with_suffix(".png"), dpi=150)
    plt.close()


def plot_attention_heatmap(weights: np.ndarray, output_path: Path) -> None:
    if weights.ndim == 4:
        attn = weights.mean(axis=0).mean(axis=0)
    elif weights.ndim == 3:
        attn = weights.mean(axis=0)
    else:
        attn = weights
    fig = go.Figure(data=go.Heatmap(z=attn, colorscale="Viridis"))
    fig.update_layout(title="Causal Attention", xaxis_title="Key", yaxis_title="Query")
    _save_plotly(fig, output_path)
    plt.figure(figsize=(7, 6))
    plt.imshow(attn, aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path.with_suffix(".png"), dpi=150)
    plt.close()


def plot_ablation_results(metrics: dict[str, object], output_path: Path) -> None:
    names = list(metrics.keys())
    ood = [float(metrics[n]["ood_rmse"]) for n in names]  # type: ignore[index]
    fig = go.Figure(data=[go.Bar(x=names, y=ood)])
    fig.update_layout(title="Ablation (OOD RMSE)", yaxis_title="RMSE")
    _save_plotly(fig, output_path)
    plt.figure(figsize=(10, 5))
    plt.bar(names, ood)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("OOD RMSE")
    plt.tight_layout()
    plt.savefig(output_path.with_suffix(".png"), dpi=150)
    plt.close()
