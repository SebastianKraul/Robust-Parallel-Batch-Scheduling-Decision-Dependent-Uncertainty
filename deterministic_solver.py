"""Deterministic arc-flow solver (Incom-P-Flow-VI) and buffer heuristic."""

import logging
import math
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from src.helpers import (
    calculate_bin_packing_lower_bound,
    calculate_ofq_threshold,
    find_reachable_nodes,
)
from src.reconstruction import reconstruct_schedule

logger = logging.getLogger(__name__)


def precompute_data(
    jobs: dict[int, tuple[int, int, int]],
    machine_capacity: int,
) -> tuple:
    """Precompute sets, parameters, and arc-flow graph data.

    Args:
        jobs: Mapping job_id -> (processing_time, size, family).
        machine_capacity: Batch capacity B.

    Returns:
        Tuple (F, all_job_sizes, Tf, Pf, N_ft, NT_sft, N_sf_le_t,
               Ofq_thresholds, V_ft, A_J_ft, A_L_ft).
    """
    F = sorted(set(fam for _, _, fam in jobs.values()))
    all_job_sizes = set(s for _, s, _ in jobs.values())
    family_p_times = {
        g: sorted(set(p for p, s, fam in jobs.values() if fam == g))
        for g in F
    }
    Tf: dict[int, list[int]] = {
        g: list(range(1, len(family_p_times[g]) + 1)) for g in F
    }
    Pf: dict[int, dict[int, int]] = {
        g: {t: p_time for t, p_time in enumerate(family_p_times[g], 1)}
        for g in F
    }
    N_ft = {
        (g, t): sum(1 for p, s, fam in jobs.values() if fam == g and p == Pf[g][t])
        for g in F for t in Tf[g]
    }
    NT_sft = {
        (g, s, t): sum(
            1 for p, sz, fam in jobs.values()
            if fam == g and sz == s and p == Pf[g][t]
        )
        for g in F for s in all_job_sizes for t in Tf[g]
    }
    N_sf_le_t = {
        (g, s, t): sum(
            1 for p, sz, fam in jobs.values()
            if fam == g and sz == s and p <= Pf[g][t]
        )
        for g in F for s in all_job_sizes for t in Tf[g]
    }
    Ofq_thresholds = {
        g: calculate_ofq_threshold(
            sorted(set(s for p, s, fam in jobs.values() if fam == g)),
            machine_capacity,
        )
        for g in F
    }
    V_ft: dict[tuple[int, int], list[int]] = {}
    A_J_ft: dict[tuple[int, int], list[tuple[int, int]]] = {}
    A_L_ft: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for g in F:
        for t in Tf[g]:
            eligible_job_sizes = sorted(
                set(s for p, s, fam in jobs.values() if fam == g and p <= Pf[g][t])
            )
            if not eligible_job_sizes:
                continue
            key = (g, t)
            V_ft[key] = find_reachable_nodes(eligible_job_sizes, machine_capacity)
            A_J_ft[key] = [
                (i, j)
                for i in V_ft[key]
                for j in V_ft[key]
                if (j - i) in eligible_job_sizes and i < j
            ]
            A_L_ft[key] = [
                (i, machine_capacity)
                for i in V_ft[key]
                if 0 < i < machine_capacity
            ]
    return (
        F, all_job_sizes, Tf, Pf, N_ft, NT_sft, N_sf_le_t,
        Ofq_thresholds, V_ft, A_J_ft, A_L_ft,
    )


def solve_incom_p_flow_vi_exact_gft(
    jobs: dict[int, tuple[int, int, int]],
    machine_capacity: int,
    num_machines: int,
    time_limit: int | float = 1800,
    mip_gap_abs: float = 0.99,
) -> tuple[str, float | None, dict | None, dict | None, float | None]:
    """Solve the deterministic arc-flow model (Incom-P-Flow-VI).

    Args:
        jobs: Mapping job_id -> (processing_time, size, family).
        machine_capacity: Batch capacity B.
        num_machines: Number of parallel machines.
        time_limit: Solver time limit in seconds.
        mip_gap_abs: Absolute MIP gap tolerance.

    Returns:
        (status_str, makespan, schedule, w_solution, mip_gap).
    """
    logger.info(
        "Building deterministic model: %d jobs, %d machines, capacity %d",
        len(jobs), num_machines, machine_capacity,
    )

    data = precompute_data(jobs, machine_capacity)
    F, all_job_sizes, Tf, Pf, N_ft, NT_sft, N_sf_le_t, Ofq_thresholds, V_ft, A_J_ft, A_L_ft = data

    model = gp.Model("Incom_P_Flow_VI_Exact_G_ft")
    model.setParam("OutputFlag", 0)

    # Decision variables
    f = model.addVars(
        [(d, e, t, g) for g in F for t in Tf[g] for d, e in A_J_ft.get((g, t), [])],
        vtype=GRB.INTEGER, name="f",
    )
    y = model.addVars(
        [(d, e, t, g) for g in F for t in Tf[g] for d, e in A_L_ft.get((g, t), [])],
        vtype=GRB.INTEGER, name="y",
    )
    v = model.addVars(
        [(t, g) for g in F for t in Tf[g]], vtype=GRB.CONTINUOUS, name="v",
    )
    z = model.addVars(
        [(s, g, t) for g in F for s in all_job_sizes for t in Tf[g]],
        vtype=GRB.CONTINUOUS, name="z",
    )
    w = model.addVars(
        [(t, g, k) for g in F for t in Tf[g] for k in range(1, num_machines + 1)],
        vtype=GRB.INTEGER, name="w",
    )
    C_max = model.addVar(vtype=GRB.CONTINUOUS, name="C_max")
    model.setObjective(C_max, GRB.MINIMIZE)

    # Constraints
    logger.debug("Adding flow conservation and assignment constraints")
    for g in F:
        family_job_sizes = set(s for p, s, fam in jobs.values() if fam == g)
        for t in Tf[g]:
            key = (g, t)
            if key not in V_ft:
                continue
            # Flow out of source
            model.addConstr(
                gp.quicksum(f.get((0, j, t, g), 0) for i, j in A_J_ft[key] if i == 0)
                + gp.quicksum(y.get((0, j, t, g), 0) for i, j in A_L_ft[key] if i == 0)
                == v[t, g],
                name=f"Flow_Out_Source_t{t}_g{g}",
            )
            # Flow into sink
            model.addConstr(
                gp.quicksum(
                    f.get((i, machine_capacity, t, g), 0)
                    for i, j in A_J_ft[key] if j == machine_capacity
                )
                + gp.quicksum(
                    y.get((i, machine_capacity, t, g), 0)
                    for i, j in A_L_ft[key] if j == machine_capacity
                )
                == v[t, g],
                name=f"Flow_In_Sink_t{t}_g{g}",
            )
            # Flow conservation at intermediate nodes
            for k_node in V_ft[key]:
                if k_node != 0 and k_node != machine_capacity:
                    model.addConstr(
                        gp.quicksum(
                            f.get((i, k_node, t, g), 0)
                            for i, j in A_J_ft[key] if j == k_node
                        )
                        + gp.quicksum(
                            y.get((i, k_node, t, g), 0)
                            for i, j in A_L_ft[key] if j == k_node
                        )
                        == gp.quicksum(
                            f.get((k_node, j, t, g), 0)
                            for i, j in A_J_ft[key] if i == k_node
                        )
                        + gp.quicksum(
                            y.get((k_node, j, t, g), 0)
                            for i, j in A_L_ft[key] if i == k_node
                        ),
                        name=f"Flow_Cons_Node_{k_node}_t{t}_g{g}",
                    )
            # Job assignment constraints
            for s in family_job_sizes:
                flow_sum_for_s = gp.quicksum(
                    f.get((i, j, t, g), 0)
                    for i, j in A_J_ft.get(key, []) if (j - i) == s
                )
                if t == 1:
                    if len(Tf[g]) == 1:
                        model.addConstr(
                            NT_sft.get((g, s, 1), 0) - flow_sum_for_s == 0,
                            name=f"Job_Assign_s{s}_g{g}_t1_final",
                        )
                    else:
                        model.addConstr(
                            NT_sft.get((g, s, 1), 0) - flow_sum_for_s == z[s, g, 1],
                            name=f"Job_Assign_s{s}_g{g}_t1",
                        )
                elif t == len(Tf[g]):
                    model.addConstr(
                        NT_sft.get((g, s, t), 0) - flow_sum_for_s == -z[s, g, t - 1],
                        name=f"Job_Assign_s{s}_g{g}_final",
                    )
                else:
                    model.addConstr(
                        NT_sft.get((g, s, t), 0) - flow_sum_for_s
                        == z[s, g, t] - z[s, g, t - 1],
                        name=f"Job_Assign_s{s}_g{g}_t{t}",
                    )
            # Machine assignment
            model.addConstr(
                gp.quicksum(w[t, g, k] for k in range(1, num_machines + 1)) == v[t, g],
                name=f"Machine_Assign_t{t}_g{g}",
            )
            # Valid inequality: bin-packing lower bound
            jobs_ge_t_sizes = [
                s for p, s, fam in jobs.values() if fam == g and p >= Pf[g][t]
            ]
            lower_bound = calculate_bin_packing_lower_bound(
                jobs_ge_t_sizes, machine_capacity
            )
            model.addConstr(
                gp.quicksum(v[t_prime, g] for t_prime in Tf[g] if t_prime >= t)
                >= lower_bound,
                name=f"VI_LB_Bin_g{g}_t{t}",
            )
            # Variable upper bounds
            model.addConstr(
                v[t, g] <= N_ft.get((g, t), 0), name=f"UB_v_t{t}_g{g}"
            )
            for d, e in A_J_ft[key]:
                job_size = e - d
                bound = min(
                    N_ft.get((g, t), 0), N_sf_le_t.get((g, job_size, t), 0)
                )
                model.addConstr(
                    f[d, e, t, g] <= bound, name=f"UB_f_{d}_{e}_t{t}_g{g}"
                )
            for d, e in A_L_ft[key]:
                model.addConstr(
                    y[d, e, t, g] <= N_ft.get((g, t), 0),
                    name=f"UB_y_{d}_{e}_t{t}_g{g}",
                )
            for s in family_job_sizes:
                if t < len(Tf[g]):
                    model.addConstr(
                        z[s, g, t] <= N_sf_le_t.get((g, s, t), 0),
                        name=f"UB_z_s{s}_g{g}_t{t}",
                    )
        # Valid inequality: non-full batch
        threshold = Ofq_thresholds[g]
        model.addConstr(
            gp.quicksum(
                y.get((d, e, t, g), 0)
                for t in Tf[g]
                for d, e in A_L_ft.get((g, t), [])
                if d > 0 and d < threshold
            )
            <= 1,
            name=f"VI_NonFull_g{g}",
        )

    # Makespan constraints
    for k in range(1, num_machines + 1):
        model.addConstr(
            gp.quicksum(Pf[g][t] * w[t, g, k] for g in F for t in Tf[g]) <= C_max,
            name=f"Makespan_Calc_Machine_{k}",
        )

    model.setParam("TimeLimit", time_limit)
    model.setParam("MIPGapAbs", mip_gap_abs)
    logger.info("Solving deterministic model")
    model.optimize()

    status = model.Status
    if model.SolCount > 0:
        status_str = "Optimal" if status == GRB.OPTIMAL else "Sub-optimal"
        makespan = C_max.X
        mip_gap = model.MIPGap
        logger.info(
            "Deterministic solve status: %s, makespan: %.2f, MIP gap: %.4f",
            status_str, makespan, mip_gap,
        )
        w_sol = model.getAttr("X", w)
        schedule = reconstruct_schedule(
            jobs, machine_capacity, num_machines, Pf, Tf, F, w_sol
        )
        return status_str, makespan, schedule, w_sol, mip_gap
    else:
        logger.warning("Deterministic solve status: Infeasible")
        return "Infeasible", None, None, None, None


def solve_buffer_heuristic(
    jobs: dict[int, tuple[int, int, int]],
    machine_capacity: int,
    num_machines: int,
    p_hat: float,
    time_limit: int | float = 1800,
    mip_gap_abs: float = 0.99,
) -> tuple[str, float | None, dict | None, dict | None, float | None]:
    """Solve the deterministic model with inflated processing times.

    Inflates each job's processing time to ceil(p_j * (1 + p_hat)).

    Args:
        jobs: Mapping job_id -> (processing_time, size, family).
        machine_capacity: Batch capacity B.
        num_machines: Number of parallel machines.
        p_hat: Uncertainty magnitude for inflation.
        time_limit: Solver time limit in seconds.
        mip_gap_abs: Absolute MIP gap tolerance.

    Returns:
        (status_str, makespan, schedule, w_solution, mip_gap).
    """
    logger.info("Solving buffer heuristic with p_hat=%.2f", p_hat)
    inflated_jobs = {
        job_id: (math.ceil(p * (1 + p_hat)), s, f)
        for job_id, (p, s, f) in jobs.items()
    }
    return solve_incom_p_flow_vi_exact_gft(
        inflated_jobs, machine_capacity, num_machines, time_limit, mip_gap_abs
    )
