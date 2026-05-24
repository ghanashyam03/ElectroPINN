"""PyTorch Lightning DataModule for battery sequences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pytorch_lightning as pl_lightning
import torch
from torch.utils.data import DataLoader, Dataset

from data.scalers import FeatureScaler
from utils.io import ensure_dir


NORM_FEATURES = ("current_n", "voltage_n", "temperature_n", "elapsed_time_n")
RAW_FEATURES = ("current", "voltage", "temperature", "elapsed_time")


@dataclass
class _SimulationCache:
    """Precomputed arrays for one simulation trajectory."""

    features: np.ndarray
    soc_n: np.ndarray
    soc_physical: np.ndarray
    dt: np.ndarray
    current: np.ndarray
    voltage: np.ndarray
    temperature: np.ndarray
    elapsed_time: np.ndarray


class BatterySequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset with per-simulation cached arrays and O(1) sequence slicing."""

    def __init__(
        self,
        parquet_path: str | Path,
        use_normalized: bool = True,
        sequence_length: int = 32,
    ) -> None:
        self.path = Path(parquet_path)
        self.use_normalized = use_normalized
        self.sequence_length = sequence_length
        self.feature_cols = list(NORM_FEATURES if use_normalized else RAW_FEATURES)
        self._caches: dict[int, _SimulationCache] = {}
        self._index: list[tuple[int, int]] = []
        self._build_caches()

    def _build_caches(self) -> None:
        df = pl.read_parquet(self.path).sort(["simulation_id", "timestamp"])
        stride = max(1, self.sequence_length // 4)

        for sim_id in df["simulation_id"].unique().to_list():
            sub = df.filter(pl.col("simulation_id") == sim_id)
            n = sub.height
            if n < self.sequence_length:
                continue

            features = sub.select(self.feature_cols).to_numpy().astype(np.float32)
            soc_n = sub["soc_n"].to_numpy().astype(np.float32)
            soc_phys = sub["soc"].to_numpy().astype(np.float32)
            ts = sub["timestamp"].to_numpy().astype(np.float32)
            elapsed = (ts - ts[0]).astype(np.float32)
            dt = np.diff(ts, prepend=ts[0]).astype(np.float32)
            current = sub["current"].to_numpy().astype(np.float32)
            voltage = sub["voltage"].to_numpy().astype(np.float32)
            temperature = sub["temperature"].to_numpy().astype(np.float32)

            self._caches[int(sim_id)] = _SimulationCache(
                features=features,
                soc_n=soc_n,
                soc_physical=soc_phys,
                dt=dt,
                current=current,
                voltage=voltage,
                temperature=temperature,
                elapsed_time=elapsed,
            )

            for start in range(0, n - self.sequence_length + 1, stride):
                self._index.append((int(sim_id), start))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sim_id, start = self._index[idx]
        cache = self._caches[sim_id]
        end = start + self.sequence_length
        sl = slice(start, end)

        return {
            "features": torch.from_numpy(cache.features[sl]),
            "soc": torch.from_numpy(cache.soc_n[sl]).unsqueeze(-1),
            "soc_physical": torch.from_numpy(cache.soc_physical[sl]).unsqueeze(-1),
            "dt": torch.from_numpy(cache.dt[sl]),
            "current": torch.from_numpy(cache.current[sl]),
            "voltage": torch.from_numpy(cache.voltage[sl]),
            "temperature": torch.from_numpy(cache.temperature[sl]),
            "elapsed_time": torch.from_numpy(cache.elapsed_time[sl]),
        }


class BatteryDataModule(pl_lightning.LightningDataModule):
    """Lightning DataModule with bounded workers and no GPU caching."""

    def __init__(
        self,
        processed_dir: str | Path,
        batch_size: int = 32,
        num_workers: int = 2,
        sequence_length: int = 32,
    ) -> None:
        super().__init__()
        self.processed_dir = Path(processed_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sequence_length = sequence_length
        self.train_ds: BatterySequenceDataset | None = None
        self.val_ds: BatterySequenceDataset | None = None
        self.test_ds: BatterySequenceDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        ensure_dir(self.processed_dir)
        train_path = self.processed_dir / "train.parquet"
        val_path = self.processed_dir / "val.parquet"
        test_path = self.processed_dir / "test.parquet"
        if stage in ("fit", None) and train_path.exists():
            self.train_ds = BatterySequenceDataset(train_path, sequence_length=self.sequence_length)
            self.val_ds = BatterySequenceDataset(val_path, sequence_length=self.sequence_length)
        if stage in ("test", "predict", None) and test_path.exists():
            self.test_ds = BatterySequenceDataset(test_path, sequence_length=self.sequence_length)

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_ds is not None
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    @property
    def scaler(self) -> FeatureScaler | None:
        scalers_path = self.processed_dir / "scalers"
        if (scalers_path / "scalers.json").exists():
            return FeatureScaler.load(scalers_path)
        alt = self.processed_dir.parent / "processed" / "scalers"
        if (alt / "scalers.json").exists():
            return FeatureScaler.load(alt)
        return None
