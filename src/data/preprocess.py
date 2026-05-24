"""Preprocess raw Parquet simulations into train/val/test splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from data.scalers import FEATURE_COLUMNS, TARGET_COLUMN, FeatureScaler
from utils.io import ensure_dir, save_json


def add_elapsed_time(df: pl.DataFrame) -> pl.DataFrame:
    """Add per-simulation elapsed time feature."""
    return df.with_columns(
        (pl.col("timestamp") - pl.col("timestamp").min().over("simulation_id")).alias(
            "elapsed_time"
        )
    )


def split_by_simulation(
    df: pl.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split simulations (not rows) into train/val/test."""
    sim_ids = df.select("simulation_id").unique().to_series().to_list()
    rng = np.random.default_rng(seed)
    rng.shuffle(sim_ids)
    n = len(sim_ids)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio)) if n > 2 else 0
    n_train = min(n_train, n - 1) if n > 1 else 1
    train_ids = set(sim_ids[:n_train])
    val_ids = set(sim_ids[n_train : n_train + n_val]) if n_val > 0 else set()
    test_ids = set(sim_ids[n_train + n_val :]) or {sim_ids[-1]}

    train_df = df.filter(pl.col("simulation_id").is_in(list(train_ids)))
    val_df = df.filter(pl.col("simulation_id").is_in(list(val_ids)))
    test_df = df.filter(pl.col("simulation_id").is_in(list(test_ids)))
    return train_df, val_df, test_df


def split_by_profile(
    df: pl.DataFrame,
    train_profiles: list[str],
    test_profiles: list[str],
    val_ratio: float,
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Profile-based split for generalization experiments."""
    train_pool = df.filter(pl.col("profile_type").is_in(train_profiles))
    test_df = df.filter(pl.col("profile_type").is_in(test_profiles))

    sim_ids = train_pool.select("simulation_id").unique().to_series().to_list()
    rng = np.random.default_rng(seed)
    rng.shuffle(sim_ids)
    n_val = max(1, int(len(sim_ids) * val_ratio))
    val_ids = set(sim_ids[:n_val])
    train_ids = set(sim_ids[n_val:])

    train_df = train_pool.filter(pl.col("simulation_id").is_in(list(train_ids)))
    val_df = train_pool.filter(pl.col("simulation_id").is_in(list(val_ids)))
    return train_df, val_df, test_df


def preprocess_pipeline(
    raw_path: Path,
    processed_dir: Path,
    scalers_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    train_profiles: list[str] | None,
    test_profiles: list[str] | None,
    seed: int,
) -> dict[str, str]:
    """Full preprocessing: features, split, scale, persist."""
    ensure_dir(processed_dir)
    ensure_dir(scalers_dir)

    df = pl.read_parquet(raw_path)
    df = add_elapsed_time(df)

    if train_profiles and test_profiles:
        train_df, val_df, test_df = split_by_profile(
            df, train_profiles, test_profiles, val_ratio=val_ratio, seed=seed
        )
    else:
        train_df, val_df, test_df = split_by_simulation(
            df, train_ratio, val_ratio, test_ratio, seed=seed
        )

    scaler = FeatureScaler.fit_from_frame(train_df)

    def _transform(split: pl.DataFrame) -> pl.DataFrame:
        x = split.select(list(FEATURE_COLUMNS)).to_numpy()
        y = split.select([TARGET_COLUMN]).to_numpy()
        x_s = scaler.transform_features(x)
        y_s = scaler.transform_target(y)
        out = split.with_columns(
            [
                pl.Series("current_n", x_s[:, 0]),
                pl.Series("voltage_n", x_s[:, 1]),
                pl.Series("temperature_n", x_s[:, 2]),
                pl.Series("elapsed_time_n", x_s[:, 3]),
                pl.Series("soc_n", y_s.ravel()),
            ]
        )
        return out

    train_t = _transform(train_df)
    val_t = _transform(val_df) if val_df.height > 0 else train_t.head(0)
    test_t = _transform(test_df) if test_df.height > 0 else train_t.head(0)

    train_t.write_parquet(processed_dir / "train.parquet")
    val_t.write_parquet(processed_dir / "val.parquet")
    test_t.write_parquet(processed_dir / "test.parquet")
    scaler.save(scalers_dir)

    meta = {
        "train_rows": train_t.height,
        "val_rows": val_t.height,
        "test_rows": test_t.height,
        "feature_columns": list(FEATURE_COLUMNS),
        "normalized_columns": [
            "current_n",
            "voltage_n",
            "temperature_n",
            "elapsed_time_n",
            "soc_n",
        ],
        "train_profiles": train_profiles,
        "test_profiles": test_profiles,
    }
    save_json(processed_dir / "preprocess_metadata.json", meta)
    return {
        "train": str(processed_dir / "train.parquet"),
        "val": str(processed_dir / "val.parquet"),
        "test": str(processed_dir / "test.parquet"),
    }
