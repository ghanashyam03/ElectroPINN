"""PyBaMM parameter set loading and overrides."""

from __future__ import annotations

from typing import Any

import pybamm


def load_parameter_values(
    name: str = "Chen2020",
    ambient_temperature_c: float = 25.0,
) -> pybamm.ParameterValues:
    """Load a PyBaMM parameter set with ambient temperature override."""
    params = pybamm.ParameterValues(name)
    params.update(
        {
            "Ambient temperature [K]": ambient_temperature_c + 273.15,
            "Initial temperature [K]": ambient_temperature_c + 273.15,
        },
        check_already_exists=False,
    )
    return params


def get_nominal_capacity_ah(params: pybamm.ParameterValues) -> float:
    """Extract nominal capacity in Ah from parameter values."""
    for key in (
        "Nominal cell capacity [A.h]",
        "Cell capacity [A.h]",
    ):
        try:
            val = params[key]
            return float(val)
        except KeyError:
            continue
    return 5.0


def simulation_inputs(ambient_temperature_c: float) -> dict[str, Any]:
    """Runtime inputs for temperature-conditioned simulations."""
    return {"Ambient temperature [K]": ambient_temperature_c + 273.15}
