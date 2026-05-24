"""Tests for PyBaMM simulation and current profiles."""

from __future__ import annotations

import numpy as np
import pytest

from simulation.current_profiles import ProfileType, generate_current_profile
from simulation.pybamm_runner import SimulationConfig, run_single_simulation


@pytest.mark.parametrize(
    "profile",
    [
        ProfileType.CONSTANT,
        ProfileType.PULSE,
        ProfileType.RANDOM,
        ProfileType.WLTP,
    ],
)
def test_current_profiles_deterministic(profile: ProfileType) -> None:
    t1, i1 = generate_current_profile(profile.value, 100.0, 5.0, 0.5, 3.0, seed=7)
    t2, i2 = generate_current_profile(profile.value, 100.0, 5.0, 0.5, 3.0, seed=7)
    np.testing.assert_allclose(t1, t2)
    np.testing.assert_allclose(i1, i2)
    assert len(t1) >= 2
    assert np.all(i1 >= 0)


@pytest.mark.slow
def test_pybamm_single_simulation() -> None:
    """Run a minimal PyBaMM SPM simulation (skipped in fast CI via marker)."""
    cfg = SimulationConfig(
        simulation_id=0,
        profile_type="constant",
        duration_s=120.0,
        timestep_s=30.0,
        current_min_a=0.5,
        current_max_a=2.0,
        c_rate=0.5,
        ambient_temperature_c=25.0,
        seed=0,
    )
    df = run_single_simulation(cfg)
    required = {
        "timestamp",
        "current",
        "voltage",
        "soc",
        "temperature",
        "ambient_temperature",
        "profile_type",
        "simulation_id",
    }
    assert required.issubset(df.columns)
    assert len(df) > 5
    assert df["soc"].between(0, 1).all()
    assert df["voltage"].between(2.0, 4.5).all()
