"""Deterministic, vectorized battery current profiles."""

from __future__ import annotations

from enum import Enum

import numpy as np


class ProfileType(str, Enum):
    """Supported drive-cycle profile types."""

    CONSTANT = "constant"
    PULSE = "pulse"
    RANDOM = "random"
    WLTP = "wltp"


def _time_grid(duration_s: float, timestep_s: float) -> np.ndarray:
    n_steps = max(int(np.ceil(duration_s / timestep_s)) + 1, 2)
    t = np.linspace(0.0, duration_s, n_steps, dtype=np.float64)
    return t


def constant_discharge(
    duration_s: float,
    timestep_s: float,
    current_a: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Constant discharge current (positive = discharge)."""
    t = _time_grid(duration_s, timestep_s)
    i = np.full_like(t, fill_value=abs(current_a))
    return t, i


def pulse_discharge(
    duration_s: float,
    timestep_s: float,
    current_min_a: float,
    current_max_a: float,
    seed: int,
    pulse_period_s: float = 120.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Square-wave pulse discharge between min and max current."""
    rng = np.random.default_rng(seed)
    t = _time_grid(duration_s, timestep_s)
    phase = (t % pulse_period_s) / pulse_period_s
    base = np.where(phase < 0.5, current_max_a, current_min_a)
    jitter = rng.normal(0.0, 0.02 * current_max_a, size=t.shape)
    i = np.clip(base + jitter, current_min_a, current_max_a)
    return t, i


def random_dynamic(
    duration_s: float,
    timestep_s: float,
    current_min_a: float,
    current_max_a: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed random walk current profile."""
    rng = np.random.default_rng(seed)
    t = _time_grid(duration_s, timestep_s)
    n = len(t)
    noise = rng.normal(0.0, 1.0, size=n)
    window = min(15, n if n % 2 == 1 else max(n - 1, 3))
    window = max(window, 3)
    kernel = np.ones(window) / float(window)
    smooth = np.convolve(noise, kernel, mode="same")
    if smooth.shape[0] != n:
        smooth = smooth[:n]
    i = current_min_a + (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-8)
    i = i * (current_max_a - current_min_a) + current_min_a
    return t, np.clip(i, current_min_a, current_max_a)


def wltp_inspired(
    duration_s: float,
    timestep_s: float,
    current_min_a: float,
    current_max_a: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """WLTP-inspired synthetic EV load (deterministic harmonics + envelope)."""
    rng = np.random.default_rng(seed)
    t = _time_grid(duration_s, timestep_s)
    t_norm = t / max(duration_s, 1.0)
    # Urban / extra-urban / highway style envelope
    env = (
        0.35 * np.sin(2 * np.pi * 0.8 * t_norm)
        + 0.25 * np.sin(2 * np.pi * 2.3 * t_norm + 0.5)
        + 0.20 * np.sin(2 * np.pi * 5.1 * t_norm + 1.1)
        + 0.20 * (t_norm**1.5)
    )
    env = (env - env.min()) / (env.max() - env.min() + 1e-8)
    micro = 0.05 * rng.normal(size=t.shape)
    i = current_min_a + (env + micro) * (current_max_a - current_min_a)
    return t, np.clip(i, current_min_a, current_max_a)


def generate_current_profile(
    profile_type: str | ProfileType,
    duration_s: float,
    timestep_s: float,
    current_min_a: float,
    current_max_a: float,
    seed: int,
    nominal_current_a: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (time_s, current_a) for a named profile."""
    ptype = ProfileType(profile_type)
    i_nom = nominal_current_a or 0.5 * (current_min_a + current_max_a)

    if ptype == ProfileType.CONSTANT:
        return constant_discharge(duration_s, timestep_s, i_nom)
    if ptype == ProfileType.PULSE:
        return pulse_discharge(
            duration_s, timestep_s, current_min_a, current_max_a, seed=seed
        )
    if ptype == ProfileType.RANDOM:
        return random_dynamic(
            duration_s, timestep_s, current_min_a, current_max_a, seed=seed
        )
    if ptype == ProfileType.WLTP:
        return wltp_inspired(
            duration_s, timestep_s, current_min_a, current_max_a, seed=seed
        )
    raise ValueError(f"Unknown profile type: {profile_type}")
