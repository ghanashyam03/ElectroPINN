"""Generate PyBaMM simulation datasets stored as Parquet."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import polars as pl
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field

from data.preprocess import preprocess_pipeline
from data.validation import validate_simulation_dataframe
from simulation.pybamm_runner import generate_simulation_grid, run_single_simulation
from utils.hydra_config import load_config, project_root
from utils.io import ensure_dir, save_json
from utils.logging_utils import get_logger, setup_logging
from utils.seed import set_seed

logger = get_logger(__name__)


class SimulationSettings(BaseModel):
    """Validated simulation configuration."""

    num_simulations: int = Field(ge=1, le=500)
    duration_s: float = Field(gt=0)
    timestep_s: float = Field(gt=0)
    current_min_a: float = Field(gt=0)
    current_max_a: float = Field(gt=0)
    c_rates: list[float]
    ambient_temperatures_c: list[float]
    parameter_set: str
    profiles: list[str]
    seed: int
    raw_data_dir: str
    fast_test: bool = False
    fast_test_simulations: int = 3
    fast_test_duration_s: float = 60.0


def main(cfg: DictConfig) -> None:
    """Entry point for dataset generation."""
    setup_logging()
    os.chdir(project_root())

    sim_cfg = OmegaConf.to_container(cfg.simulation, resolve=True)
    assert isinstance(sim_cfg, dict)
    raw_dir = Path(cfg.paths.raw_data)
    ensure_dir(raw_dir)

    fast_test = os.environ.get("BATTERY_PINN_FAST_TEST", "0") == "1"
    settings = SimulationSettings(
        **sim_cfg,
        raw_data_dir=str(raw_dir),
        fast_test=fast_test,
    )

    set_seed(int(cfg.project.seed), deterministic=bool(cfg.project.deterministic))

    num_sims = settings.fast_test_simulations if settings.fast_test else settings.num_simulations
    duration = settings.fast_test_duration_s if settings.fast_test else settings.duration_s

    logger.info(
        "Generating %s simulations | duration=%.0fs | timestep=%.0fs | fast_test=%s",
        num_sims,
        duration,
        settings.timestep_s,
        fast_test,
    )
    logger.info(
        "First PyBaMM solve may take 30-90s on CPU while CasADi initializes; progress updates after each run."
    )

    configs = generate_simulation_grid(
        num_simulations=num_sims,
        profiles=settings.profiles,
        c_rates=settings.c_rates,
        ambient_temperatures_c=settings.ambient_temperatures_c,
        duration_s=duration,
        timestep_s=settings.timestep_s,
        current_min_a=settings.current_min_a,
        current_max_a=settings.current_max_a,
        parameter_set=settings.parameter_set,
        seed=settings.seed,
    )

    frames: list[pl.DataFrame] = []
    failed = 0
    total = len(configs)
    for idx, sim_config in enumerate(configs):
        logger.info(
            "Simulation %s/%s | id=%s profile=%s",
            idx + 1,
            total,
            sim_config.simulation_id,
            sim_config.profile_type,
        )
        try:
            pdf = run_single_simulation(sim_config)
            if validate_simulation_dataframe(pdf, sim_config.simulation_id):
                frames.append(pl.from_pandas(pdf))
            else:
                failed += 1
                logger.warning("Simulation %s rejected by validation", sim_config.simulation_id)
        except Exception as exc:
            failed += 1
            logger.error("Simulation %s failed: %s", sim_config.simulation_id, exc)
            if settings.fast_test:
                raise

    if not frames:
        logger.error("No simulations succeeded")
        sys.exit(1)

    combined = pl.concat(frames, how="vertical_relaxed")
    out_path = raw_dir / "simulations.parquet"
    combined.write_parquet(out_path, compression="zstd")

    metadata = {
        "num_simulations": num_sims,
        "num_rows": combined.height,
        "failed": failed,
        "columns": combined.columns,
        "settings": settings.model_dump(),
    }
    save_json(raw_dir / "metadata.json", metadata)
    logger.info("Saved %s rows to %s (%s failures)", combined.height, out_path, failed)

    processed_dir = Path(cfg.paths.processed_data)
    scalers_dir = Path(cfg.paths.scalers)
    preprocess_pipeline(
        raw_path=out_path,
        processed_dir=processed_dir,
        scalers_dir=scalers_dir,
        train_ratio=float(cfg.train.split.train_ratio),
        val_ratio=float(cfg.train.split.val_ratio),
        test_ratio=float(cfg.train.split.test_ratio),
        train_profiles=list(cfg.train.train_profiles),
        test_profiles=list(cfg.train.test_profiles),
        seed=int(cfg.project.seed),
    )
    logger.info("Preprocessed dataset written to %s", processed_dir)


if __name__ == "__main__":
    main(load_config())
