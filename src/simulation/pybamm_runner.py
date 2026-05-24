"""PyBaMM SPM simulation runner with custom current profiles."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pybamm

from simulation.current_profiles import ProfileType, generate_current_profile
from simulation.parameter_sets import (
    get_nominal_capacity_ah,
    load_parameter_values,
    simulation_inputs,
)
from utils.logging_utils import get_logger

logger = get_logger(__name__)

VOLTAGE_KEY = "Terminal voltage [V]"
CURRENT_KEY = "Current [A]"
TIME_KEY = "Time [s]"
TEMP_KEYS = (
    "X-averaged cell temperature [K]",
    "Cell temperature [K]",
    "Volume-averaged cell temperature [K]",
)


def _build_solver() -> pybamm.BaseSolver:
    """Prefer fast CasADi; fall back to SciPy if unavailable."""
    try:
        return pybamm.CasadiSolver(mode="fast", atol=1e-6, rtol=1e-6)
    except Exception:
        logger.warning("CasadiSolver unavailable; using ScipySolver")
        return pybamm.ScipySolver()


def _build_model() -> pybamm.BaseModel:
    """SPM with isothermal thermal mode for numerical stability on laptop CPUs."""
    return pybamm.lithium_ion.SPM({"thermal": "isothermal"})


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for a single PyBaMM simulation."""

    simulation_id: int
    profile_type: str
    duration_s: float
    timestep_s: float
    current_min_a: float
    current_max_a: float
    c_rate: float
    ambient_temperature_c: float
    parameter_set: str = "Chen2020"
    seed: int = 42


def _resolve_soc(solution: pybamm.Solution, nominal_capacity_ah: float) -> np.ndarray:
    """Extract or compute state of charge in [0, 1]."""
    for key in ("State of Charge", "SOC", "State of charge"):
        try:
            soc = np.asarray(solution[key].entries, dtype=np.float64)
            if soc.max() > 1.5:
                soc = soc / 100.0
            return np.clip(soc, 0.0, 1.0)
        except (KeyError, TypeError):
            continue

    try:
        discharge_q = np.asarray(
            solution["Discharge capacity [A.h]"].entries, dtype=np.float64
        )
        soc = 1.0 - discharge_q / max(nominal_capacity_ah, 1e-6)
        return np.clip(soc, 0.0, 1.0)
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Could not resolve SOC from PyBaMM solution") from exc


def _resolve_temperature(solution: pybamm.Solution, ambient_c: float) -> np.ndarray:
    """Extract cell temperature in Celsius."""
    for key in TEMP_KEYS:
        try:
            temp_k = np.asarray(solution[key].entries, dtype=np.float64)
            return temp_k - 273.15
        except (KeyError, TypeError):
            continue
    n = len(solution[TIME_KEY].entries)
    logger.warning("Temperature variable not found; using ambient temperature")
    return np.full(n, ambient_c, dtype=np.float64)


def _build_experiment(time_s: np.ndarray, current_a: np.ndarray) -> pybamm.Experiment:
    """Build PyBaMM experiment from time-current drive cycle (Nx2 array)."""
    drive_cycle = np.column_stack([time_s, current_a])
    return pybamm.Experiment([pybamm.step.current(drive_cycle)])


def _resample_to_grid(
    t_sim: np.ndarray,
    arrays: dict[str, np.ndarray],
    t_target: np.ndarray,
) -> dict[str, np.ndarray]:
    """Linear resample simulation outputs onto the drive-cycle time grid."""
    out: dict[str, np.ndarray] = {}
    for name, values in arrays.items():
        v = np.asarray(values, dtype=np.float64).ravel()
        if len(v) != len(t_sim):
            n = min(len(v), len(t_sim))
            v = v[:n]
            t_use = t_sim[:n]
        else:
            t_use = t_sim
        if len(t_use) < 2:
            out[name] = np.full(len(t_target), v[0] if len(v) else 0.0)
            continue
        out[name] = np.interp(t_target, t_use, v)
    return out


def run_single_simulation(config: SimulationConfig) -> pd.DataFrame:
    """Run one SPM simulation and return a standardized DataFrame."""
    time_s, current_a = generate_current_profile(
        profile_type=config.profile_type,
        duration_s=config.duration_s,
        timestep_s=config.timestep_s,
        current_min_a=config.current_min_a,
        current_max_a=config.current_max_a,
        seed=config.seed + config.simulation_id,
        nominal_current_a=config.c_rate * 5.0,
    )

    param_values = load_parameter_values(
        name=config.parameter_set,
        ambient_temperature_c=config.ambient_temperature_c,
    )
    q_nom = get_nominal_capacity_ah(param_values)
    scale = config.c_rate * q_nom / max(float(np.mean(current_a)), 1e-6)
    current_a = np.clip(current_a * scale, 0.05, 15.0)

    model = _build_model()
    experiment = _build_experiment(time_s, current_a)
    inputs: dict[str, Any] = simulation_inputs(config.ambient_temperature_c)
    solver = _build_solver()

    logger.info(
        "PyBaMM solve start | id=%s profile=%s C=%.2f T_amb=%.1fC duration=%.0fs",
        config.simulation_id,
        config.profile_type,
        config.c_rate,
        config.ambient_temperature_c,
        config.duration_s,
    )
    t0 = time.perf_counter()

    sim = pybamm.Simulation(
        model,
        experiment=experiment,
        parameter_values=param_values,
        solver=solver,
    )
    solution = sim.solve(inputs=inputs, showprogress=False)

    elapsed = time.perf_counter() - t0
    logger.info(
        "PyBaMM solve done | id=%s elapsed=%.1fs",
        config.simulation_id,
        elapsed,
    )

    t_sim = np.asarray(solution[TIME_KEY].entries, dtype=np.float64).ravel()
    resampled = _resample_to_grid(
        t_sim,
        {
            "voltage": solution[VOLTAGE_KEY].entries,
            "current": solution[CURRENT_KEY].entries,
            "soc": _resolve_soc(solution, q_nom),
            "temperature": _resolve_temperature(solution, config.ambient_temperature_c),
        },
        time_s,
    )

    n = len(time_s)
    df = pd.DataFrame(
        {
            "timestamp": time_s[:n],
            "current": np.abs(resampled["current"][:n]),
            "voltage": resampled["voltage"][:n],
            "soc": resampled["soc"][:n],
            "temperature": resampled["temperature"][:n],
            "ambient_temperature": np.full(n, config.ambient_temperature_c),
            "profile_type": config.profile_type,
            "simulation_id": config.simulation_id,
        }
    )
    return df


def generate_simulation_grid(
    num_simulations: int,
    profiles: list[str],
    c_rates: list[float],
    ambient_temperatures_c: list[float],
    duration_s: float,
    timestep_s: float,
    current_min_a: float,
    current_max_a: float,
    parameter_set: str,
    seed: int,
) -> list[SimulationConfig]:
    """Build a deterministic list of simulation configurations."""
    configs: list[SimulationConfig] = []
    sim_id = 0
    rng = np.random.default_rng(seed)
    profile_cycle = [ProfileType(p).value for p in profiles]

    while len(configs) < num_simulations:
        profile = profile_cycle[sim_id % len(profile_cycle)]
        c_rate = float(c_rates[sim_id % len(c_rates)])
        t_amb = float(ambient_temperatures_c[sim_id % len(ambient_temperatures_c)])
        configs.append(
            SimulationConfig(
                simulation_id=sim_id,
                profile_type=profile,
                duration_s=duration_s,
                timestep_s=timestep_s,
                current_min_a=current_min_a,
                current_max_a=current_max_a,
                c_rate=c_rate,
                ambient_temperature_c=t_amb,
                parameter_set=parameter_set,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
        )
        sim_id += 1
    return configs
