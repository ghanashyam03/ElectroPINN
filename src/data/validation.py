"""Validation of PyBaMM simulation outputs before dataset ingestion."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.logging_utils import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = (
    "timestamp",
    "current",
    "voltage",
    "soc",
    "temperature",
    "ambient_temperature",
    "profile_type",
    "simulation_id",
)


def validate_simulation_dataframe(df: pd.DataFrame, simulation_id: int) -> bool:
    """
    Validate a single simulation DataFrame.

    Returns True if valid; False if it should be rejected.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        logger.warning("Simulation %s rejected: missing columns %s", simulation_id, missing)
        return False

    if df.empty:
        logger.warning("Simulation %s rejected: empty", simulation_id)
        return False

    if df.isnull().any().any():
        logger.warning("Simulation %s rejected: contains NaN", simulation_id)
        return False

    ts = df["timestamp"].to_numpy()
    if not np.all(np.diff(ts) >= 0):
        logger.warning("Simulation %s rejected: non-monotonic timestamps", simulation_id)
        return False

    if len(np.unique(ts)) != len(ts):
        logger.warning("Simulation %s rejected: duplicate timestamps", simulation_id)
        return False

    soc = df["soc"].to_numpy()
    if soc.min() < -0.05 or soc.max() > 1.05:
        logger.warning("Simulation %s rejected: SoC out of bounds", simulation_id)
        return False

    voltage = df["voltage"].to_numpy()
    if voltage.min() < 2.0 or voltage.max() > 4.5:
        logger.warning("Simulation %s rejected: voltage out of bounds", simulation_id)
        return False

    if not np.isfinite(df.select_dtypes(include=[np.number]).to_numpy()).all():
        logger.warning("Simulation %s rejected: non-finite values", simulation_id)
        return False

    return True
