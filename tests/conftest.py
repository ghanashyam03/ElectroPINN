"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def synthetic_parquet(tmp_path: Path) -> Path:
    """Create a small synthetic dataset for unit tests."""
    n_sims = 4
    rows_per_sim = 40
    frames: list[pl.DataFrame] = []
    for sim_id in range(n_sims):
        t = np.linspace(0, 100, rows_per_sim)
        soc = np.linspace(1.0, 0.2, rows_per_sim)
        frames.append(
            pl.DataFrame(
                {
                    "timestamp": t,
                    "current": np.full(rows_per_sim, 2.0),
                    "voltage": np.linspace(4.1, 3.2, rows_per_sim),
                    "soc": soc,
                    "temperature": np.linspace(25, 30, rows_per_sim),
                    "ambient_temperature": np.full(rows_per_sim, 25.0),
                    "profile_type": ["constant", "pulse", "random", "wltp"][sim_id % 4],
                    "simulation_id": np.full(rows_per_sim, sim_id),
                }
            )
        )
    df = pl.concat(frames)
    path = tmp_path / "simulations.parquet"
    df.write_parquet(path)
    return path
