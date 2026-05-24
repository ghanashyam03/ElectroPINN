"""Data validation tests."""

from __future__ import annotations

import pandas as pd

from data.validation import validate_simulation_dataframe


def test_validate_rejects_nan() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [0.0, 1.0],
            "current": [1.0, 1.0],
            "voltage": [3.7, 3.6],
            "soc": [1.0, float("nan")],
            "temperature": [25.0, 25.0],
            "ambient_temperature": [25.0, 25.0],
            "profile_type": ["c", "c"],
            "simulation_id": [0, 0],
        }
    )
    assert not validate_simulation_dataframe(df, 0)


def test_validate_accepts_good() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [0.0, 1.0, 2.0],
            "current": [1.0, 1.0, 1.0],
            "voltage": [3.9, 3.8, 3.7],
            "soc": [1.0, 0.9, 0.8],
            "temperature": [25.0, 25.0, 25.0],
            "ambient_temperature": [25.0, 25.0, 25.0],
            "profile_type": ["c", "c", "c"],
            "simulation_id": [0, 0, 0],
        }
    )
    assert validate_simulation_dataframe(df, 0)
