"""Configuration loading and validation for experimental parameters."""

from dataclasses import dataclass
from typing import Any

import yaml


_REQUIRED_KEYS = [
    "size_categories",
    "ptime_categories",
    "jobs",
    "machines",
    "families",
    "size_cats",
    "ptime_cats",
    "batch_capacities",
    "run_ids",
    "proportion",
    "hat",
    "solver_time_limit",
    "solver_mip_gap_abs",
]


@dataclass(frozen=True)
class ExperimentConfig:
    """Typed container for all experimental parameters."""

    size_categories: dict[str, tuple[int, int]]
    ptime_categories: dict[str, tuple[int, int]]
    jobs: list[int]
    machines: list[int]
    families: list[int]
    size_cats: list[str]
    ptime_cats: list[str]
    batch_capacities: list[int]
    run_ids: list[int]
    proportion: list[float]
    hat: list[float]
    solver_time_limit: int | float
    solver_mip_gap_abs: float


def load_config(path: str) -> ExperimentConfig:
    """Load and validate a YAML config file into an ExperimentConfig.

    Args:
        path: Path to a YAML configuration file.

    Returns:
        A validated ExperimentConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If a required key is missing.
    """
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    for key in _REQUIRED_KEYS:
        if key not in raw:
            raise ValueError(
                f"Missing required configuration key: '{key}'"
            )

    # Convert category dicts from list values [lo, hi] to tuple values (lo, hi)
    size_categories = {
        k: tuple(v) for k, v in raw["size_categories"].items()
    }
    ptime_categories = {
        k: tuple(v) for k, v in raw["ptime_categories"].items()
    }

    return ExperimentConfig(
        size_categories=size_categories,
        ptime_categories=ptime_categories,
        jobs=raw["jobs"],
        machines=raw["machines"],
        families=raw["families"],
        size_cats=raw["size_cats"],
        ptime_cats=raw["ptime_cats"],
        batch_capacities=raw["batch_capacities"],
        run_ids=raw["run_ids"],
        proportion=[float(p) for p in raw["proportion"]],
        hat=[float(h) for h in raw["hat"]],
        solver_time_limit=raw["solver_time_limit"],
        solver_mip_gap_abs=float(raw["solver_mip_gap_abs"]),
    )
