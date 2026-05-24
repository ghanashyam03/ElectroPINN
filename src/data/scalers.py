"""Feature scaling with persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = ("current", "voltage", "temperature", "elapsed_time")
TARGET_COLUMN = "soc"


@dataclass
class FeatureScaler:
    """StandardScaler wrapper for battery features."""

    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS
    target_column: str = TARGET_COLUMN

    @classmethod
    def fit_from_frame(cls, df: pl.DataFrame) -> FeatureScaler:
        """Fit scalers on a Polars DataFrame."""
        features = df.select(list(FEATURE_COLUMNS)).to_numpy()
        targets = df.select([TARGET_COLUMN]).to_numpy()
        fs = StandardScaler()
        ts = StandardScaler()
        fs.fit(features)
        ts.fit(targets)
        return cls(feature_scaler=fs, target_scaler=ts)

    def transform_features(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.feature_scaler.transform(x), dtype=np.float32)

    def transform_target(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(self.target_scaler.transform(y), dtype=np.float32)

    def inverse_target(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(self.target_scaler.inverse_transform(y), dtype=np.float32)

    def save(self, directory: str | Path) -> None:
        """Persist scaler parameters as JSON."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "feature_mean": self.feature_scaler.mean_.tolist(),
            "feature_scale": self.feature_scaler.scale_.tolist(),
            "target_mean": self.target_scaler.mean_.tolist(),
            "target_scale": self.target_scaler.scale_.tolist(),
        }
        with (path / "scalers.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory: str | Path) -> FeatureScaler:
        """Load scalers from JSON."""
        with (Path(directory) / "scalers.json").open(encoding="utf-8") as f:
            meta = json.load(f)
        fs = StandardScaler()
        ts = StandardScaler()
        fs.mean_ = np.array(meta["feature_mean"], dtype=np.float64)
        fs.scale_ = np.array(meta["feature_scale"], dtype=np.float64)
        fs.var_ = fs.scale_**2
        fs.n_features_in_ = len(fs.mean_)
        ts.mean_ = np.array(meta["target_mean"], dtype=np.float64)
        ts.scale_ = np.array(meta["target_scale"], dtype=np.float64)
        ts.var_ = ts.scale_**2
        ts.n_features_in_ = 1
        return cls(
            feature_scaler=fs,
            target_scaler=ts,
            feature_columns=tuple(meta["feature_columns"]),
            target_column=meta["target_column"],
        )
