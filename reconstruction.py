"""Schedule reconstruction from arc-flow solution."""

import logging
from typing import Any

import gurobipy as gp
from gurobipy import GRB

logger = logging.getLogger(__name__)


def reconstruct_schedule(
    jobs: dict[int, tuple[int, int, int]],
    machine_capacity: int,
    num_machines: int,
    Pf: dict[int, dict[int, int]],
    Tf: dict[int, list[int]],
    F: list[int],
    w_solution_values: dict[tuple[int, int, int], float],
) -> dict[int, list[dict[str, Any]]] | None:
    """Reconstruct a detailed schedule from the arc-flow w variables.

    Assigns each job to a batch on a machine via a small assignment MIP.

    Args:
        jobs: Mapping job_id -> (processing_time, size, family).
        machine_capacity: Batch capacity B.
        num_machines: Number of machines.
        Pf: Mapping family -> {time_index: processing_time}.
        Tf: Mapping family -> list of time indices.
        F: Sorted list of families.
        w_solution_values: Solution values for w[t, g, k] variables.

    Returns:
        Schedule dict mapping machine_id -> list of batch dicts, or None if
        reconstruction fails.
    """
    logger.debug("Reconstructing schedule")
    batch_instances: list[dict[str, Any]] = []
    batch_counter = 1
    for g in F:
        for t in Tf[g]:
            for m in range(1, num_machines + 1):
                num_slots = int(round(w_solution_values.get((t, g, m), 0)))
                for _ in range(num_slots):
                    batch_instances.append({
                        "id": f"B{batch_counter}",
                        "machine": m,
                        "p_time": Pf[g][t],
                        "family": g,
                        "capacity": machine_capacity,
                    })
                    batch_counter += 1

    job_instances = [
        {"id": j_id, "p_time": p, "size": s, "family": fam}
        for j_id, (p, s, fam) in jobs.items()
    ]
    job_ids = [j["id"] for j in job_instances]
    job_sizes = {j["id"]: j["size"] for j in job_instances}

    recon_model = gp.Model("ReconstructionAssignment")
    recon_model.setParam("OutputFlag", 0)

    valid_pairs = [
        (j["id"], b["id"])
        for j in job_instances
        for b in batch_instances
        if j["family"] == b["family"] and j["p_time"] <= b["p_time"]
    ]
    if not valid_pairs and job_ids:
        return None

    assign_vars = recon_model.addVars(valid_pairs, vtype=GRB.BINARY, name="assign")
    recon_model.setObjective(0, GRB.MINIMIZE)
    recon_model.addConstrs(
        (assign_vars.sum(j_id, "*") == 1 for j_id in job_ids),
        name="JobAssignedOnce",
    )
    batch_ids = [b["id"] for b in batch_instances]
    recon_model.addConstrs(
        (
            gp.quicksum(
                job_sizes[j_id] * assign_vars.get((j_id, b_id), 0)
                for j_id in job_ids
            )
            <= machine_capacity
            for b_id in batch_ids
        ),
        name="BatchCapacity",
    )
    recon_model.optimize()

    if recon_model.Status != GRB.OPTIMAL:
        return None

    final_schedule: dict[int, list[dict[str, Any]]] = {
        m: [] for m in range(1, num_machines + 1)
    }
    batch_job_map: dict[str, list[int]] = {b["id"]: [] for b in batch_instances}
    batch_size_map: dict[str, int] = {b["id"]: 0 for b in batch_instances}

    for (j_id, b_id), var in assign_vars.items():
        if var.X > 0.5:
            batch_job_map[b_id].append(j_id)
            batch_size_map[b_id] += job_sizes[j_id]

    for b in batch_instances:
        if batch_job_map[b["id"]]:
            final_schedule[b["machine"]].append({
                "batch_id": b["id"],
                "processing_time": b["p_time"],
                "family": b["family"],
                "jobs": sorted(batch_job_map[b["id"]]),
                "total_size": batch_size_map[b["id"]],
            })

    return final_schedule
