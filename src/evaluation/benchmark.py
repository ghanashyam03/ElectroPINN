"""Benchmark inference latency and memory footprint."""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data.datamodule import BatterySequenceDataset
from evaluation.compare_models import _latest_checkpoint
from models.inference import measure_latency
from training.lightning_module import BaselineLitModule, PINNLitModule
from utils.hydra_config import load_config, project_root
from utils.device import get_device, get_gpu_memory_mb
from utils.io import save_json
from utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def main(cfg: DictConfig) -> None:
    setup_logging()
    os.chdir(project_root())
    device = get_device()

    test_path = Path(cfg.paths.processed_data) / "test.parquet"
    if not test_path.exists():
        raise FileNotFoundError("Processed test data missing. Run training pipeline first.")

    ds = BatterySequenceDataset(test_path)
    batch = next(iter(DataLoader(ds, batch_size=int(cfg.train.batch_size))))

    baseline = BaselineLitModule.load_from_checkpoint(
        str(_latest_checkpoint(Path(cfg.paths.checkpoints) / "baseline")),
        map_location=device,
    ).to(device)
    pinn = PINNLitModule.load_from_checkpoint(
        str(_latest_checkpoint(Path(cfg.paths.checkpoints) / "pinn")),
        map_location=device,
    ).to(device)

    x = batch["features"]
    baseline_ms = measure_latency(baseline, x, device)
    pinn_ms = measure_latency(pinn, x, device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        _ = baseline(x.to(device))
        baseline_mem = get_gpu_memory_mb()
        torch.cuda.reset_peak_memory_stats()
        _ = pinn(x.to(device))
        pinn_mem = get_gpu_memory_mb()
    else:
        baseline_mem = None
        pinn_mem = None

    report = {
        "device": str(device),
        "baseline_latency_ms": baseline_ms,
        "pinn_latency_ms": pinn_ms,
        "baseline_gpu_memory_mb": baseline_mem,
        "pinn_gpu_memory_mb": pinn_mem,
        "baseline_parameters": baseline.model.count_parameters(),
        "pinn_parameters": pinn.model.count_parameters(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out = Path(cfg.paths.outputs) / "benchmark_report.json"
    save_json(out, report)
    logger.info("Benchmark report saved to %s", out)
    logger.info("%s", report)


if __name__ == "__main__":
    main(load_config())
