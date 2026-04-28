"""Instance I/O: save, load, and generate problem instances."""

import json
import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default directory for experimental study instances
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTANCES_DIR = os.path.join(_PROJECT_DIR, "experimental_study_instances")


def save_instance_to_json(
    jobs: dict[int, tuple[int, int, int]],
    num_machines: int,
    machine_capacity: int,
    filename: str,
    instances_dir: str | None = None,
) -> None:
    """Save an instance to a JSON file.

    Args:
        jobs: Mapping job_id -> (processing_time, size, family).
        num_machines: Number of parallel machines.
        machine_capacity: Batch capacity B.
        filename: Name of the output file (relative to instances_dir).
        instances_dir: Directory to save into (default: experimental_study_instances/).
    """
    directory = instances_dir or INSTANCES_DIR
    instance_data: dict[str, Any] = {
        "jobs": jobs,
        "num_machines": num_machines,
        "machine_capacity": machine_capacity,
    }
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    with open(filepath, "w") as f:
        json.dump(instance_data, f, indent=4)
    logger.debug("Instance saved to %s", filepath)


def read_instance_from_json(
    filename: str,
    instances_dir: str | None = None,
) -> tuple[dict[int, tuple[int, int, int]], int, int]:
    """Load an instance from a JSON file.

    Args:
        filename: Name of the file (relative to instances_dir).
        instances_dir: Directory to load from (default: experimental_study_instances/).

    Returns:
        (jobs, num_machines, machine_capacity).
    """
    directory = instances_dir or INSTANCES_DIR
    filepath = os.path.join(directory, filename)
    with open(filepath) as f:
        loaded_data = json.load(f)
    jobs = {int(k): tuple(v) for k, v in loaded_data["jobs"].items()}
    num_machines = loaded_data["num_machines"]
    machine_capacity = loaded_data["machine_capacity"]
    return jobs, num_machines, machine_capacity


def generate_instance(
    n_jobs: int,
    n_machines: int,
    n_families: int,
    size_range: tuple[int, int],
    p_time_range: tuple[int, int],
    capacity: int,
    seed: int | None = None,
) -> tuple[dict[int, tuple[int, int, int]], int, int]:
    """Generate a random problem instance.

    Args:
        n_jobs: Number of jobs.
        n_machines: Number of machines.
        n_families: Number of incompatible families.
        size_range: (min_size, max_size) inclusive.
        p_time_range: (min_ptime, max_ptime) inclusive.
        capacity: Batch capacity B.
        seed: Random seed for reproducibility.

    Returns:
        (jobs, num_machines, capacity).
    """
    rng = np.random.default_rng(seed)
    jobs: dict[int, tuple[int, int, int]] = {}
    for job_id in range(1, n_jobs + 1):
        p_time = int(rng.integers(p_time_range[0], p_time_range[1] + 1))
        size = int(rng.integers(size_range[0], size_range[1] + 1))
        family = int(rng.integers(1, n_families + 1))
        jobs[job_id] = (p_time, size, family)
    return jobs, n_machines, capacity
