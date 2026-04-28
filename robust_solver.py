"""Robust solver: master model with dual embedding and adversarial subproblem."""

import logging
import math
import time
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from src.deterministic_solver import precompute_data
from src.helpers import calculate_bin_packing_lower_bound, calculate_ofq_threshold, find_reachable_nodes
from src.reconstruction import reconstruct_schedule

logger = logging.getLogger(__name__)


def build_master_model(
    jobs: dict[int, tuple[int, int, int]],
    machine_capacity: int,
    num_machines: int,
    robustness_params: dict[str, Any],
    precomputed: tuple,
    mip_gap_abs: float = 0.99,
) -> tuple:
    """Build the robust master model with exact dual embedding.

    Args:
        jobs: Mapping job_id -> (processing_time, size, family).
        machine_capacity: Batch capacity B.
        num_machines: Number of parallel machines.
        robustness_params: Dict with keys 'robustness_proportion', 'p_hat_type', 'p_hat_value'.
        precomputed: Precomputed data tuple from precompute_data().
        mip_gap_abs: Absolute MIP gap tolerance.

    Returns:
        (model, w, eta, Gamma, C_max, P_hat_f).
    """
    # Normalize robustness_proportion and p_hat_value to per-machine lists
    raw_alpha = robustness_params["robustness_proportion"]
    raw_p_hat = robustness_params["p_hat_value"]
    alpha_per_machine: list[float] = (
        raw_alpha if isinstance(raw_alpha, list) else [float(raw_alpha)] * num_machines
    )
    p_hat_per_machine: list[float] = (
        raw_p_hat if isinstance(raw_p_hat, list) else [float(raw_p_hat)] * num_machines
    )

    logger.info(
        "Building master model with robustness params: alpha=%s, p_hat=%s",
        alpha_per_machine, p_hat_per_machine,
    )
    model = gp.Model("Robust_Master_Problem")
    model.setParam("OutputFlag", 0)

    F, all_job_sizes, Tf, Pf, N_ft, NT_sft, N_sf_le_t, Ofq_thresholds, V_ft, A_J_ft, A_L_ft = precomputed

    # Build per-machine P_hat_f: P_hat_f[k][g][t]
    p_hat_type = robustness_params["p_hat_type"]
    P_hat_f: dict[int, dict[int, dict[int, float]]] = {}
    for k in range(1, num_machines + 1):
        p_hat_val = p_hat_per_machine[k - 1]
        if p_hat_type == "integer":
            P_hat_f[k] = {g: {t: p_hat_val for t in Tf[g]} for g in F}
        else:
            P_hat_f[k] = {
                g: {t: math.ceil(Pf[g][t] * p_hat_val) for t in Tf[g]} for g in F
            }

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
    eta = model.addVars(range(1, num_machines + 1), lb=0.0, name="eta")
    model.setObjective(C_max, GRB.MINIMIZE)

    # Arc-flow and assignment constraints (same as deterministic)
    for g in F:
        family_job_sizes = set(s for p, s, fam in jobs.values() if fam == g)
        for t in Tf[g]:
            key = (g, t)
            if key not in V_ft:
                continue
            model.addConstr(
                gp.quicksum(f.get((0, j, t, g), 0) for i, j in A_J_ft[key] if i == 0)
                + gp.quicksum(y.get((0, j, t, g), 0) for i, j in A_L_ft[key] if i == 0)
                == v[t, g],
                name=f"Flow_Out_Source_t{t}_g{g}",
            )
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
            model.addConstr(
                gp.quicksum(w[t, g, k] for k in range(1, num_machines + 1)) == v[t, g],
                name=f"Machine_Assign_t{t}_g{g}",
            )
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
            model.addConstr(
                v[t, g] <= N_ft.get((g, t), 0), name=f"UB_v_t{t}_g{g}"
            )
            for d, e in A_J_ft[key]:
                job_size = e - d
                bound = min(N_ft.get((g, t), 0), N_sf_le_t.get((g, job_size, t), 0))
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

    # Endogenous uncertainty budget (per-machine alpha)
    Gamma = model.addVars(
        range(1, num_machines + 1), vtype=GRB.INTEGER, lb=0, name="Gamma"
    )
    total_batch_ub = sum(N_ft.get((g, t), 0) for g in F for t in Tf[g])
    for k in range(1, num_machines + 1):
        alpha_k = alpha_per_machine[k - 1]
        model.addConstr(
            Gamma[k]
            >= alpha_k * gp.quicksum(w[t, g, k] for g in F for t in Tf[g]),
            name=f"Budget_k{k}",
        )

    # Exact dual reformulation of eta[k] >= h(w_k, Gamma_k), per-machine
    for k in range(1, num_machines + 1):
        alpha_k = alpha_per_machine[k - 1]
        P_hat_k = P_hat_f[k]

        distinct_p_hat = sorted(
            set(P_hat_k[g][t] for g in F for t in Tf[g]), reverse=True
        )
        lambda_levels = distinct_p_hat + [0]
        L = len(lambda_levels)
        Gamma_ub = (
            math.ceil(alpha_k * total_batch_ub) if total_batch_ub > 0 else 1
        )

        b_lam = model.addVars(range(L), vtype=GRB.BINARY, name=f"blam_{k}")
        model.addConstr(b_lam.sum() == 1, name=f"blam_sos_{k}")

        for l_idx, p_l in enumerate(lambda_levels):
            theta_l = model.addVar(
                lb=0, ub=Gamma_ub, name=f"theta_{k}_{l_idx}"
            )
            model.addConstr(
                theta_l <= Gamma_ub * b_lam[l_idx],
                name=f"theta_ub2_{k}_{l_idx}",
            )
            model.addConstr(theta_l <= Gamma[k], name=f"theta_ub1_{k}_{l_idx}")
            model.addConstr(
                theta_l >= Gamma[k] - Gamma_ub * (1 - b_lam[l_idx]),
                name=f"theta_lb_{k}_{l_idx}",
            )

            f_l_terms = [
                (max(0, P_hat_k[g][t] - p_l), t, g)
                for g in F
                for t in Tf[g]
                if P_hat_k[g][t] > p_l
            ]
            f_l_ub = sum(coeff * N_ft.get((g, t), 0) for coeff, t, g in f_l_terms)

            if f_l_ub > 0:
                f_l_expr = gp.quicksum(coeff * w[t, g, k] for coeff, t, g in f_l_terms)
                phi_l = model.addVar(lb=0, ub=f_l_ub, name=f"phi_{k}_{l_idx}")
                model.addConstr(
                    phi_l <= f_l_ub * b_lam[l_idx],
                    name=f"phi_ub2_{k}_{l_idx}",
                )
                model.addConstr(phi_l <= f_l_expr, name=f"phi_ub1_{k}_{l_idx}")
                model.addConstr(
                    phi_l >= f_l_expr - f_l_ub * (1 - b_lam[l_idx]),
                    name=f"phi_lb_{k}_{l_idx}",
                )
            else:
                phi_l = 0

            model.addConstr(
                eta[k] >= p_l * theta_l + phi_l, name=f"rob_{k}_{l_idx}"
            )

    # Makespan constraints (with protection)
    for k in range(1, num_machines + 1):
        nominal_time = gp.quicksum(Pf[g][t] * w[t, g, k] for g in F for t in Tf[g])
        model.addConstr(
            nominal_time + eta[k] <= C_max, name=f"Makespan_Calc_Machine_{k}"
        )

    model.setParam("MIPGapAbs", mip_gap_abs)
    return model, w, eta, Gamma, C_max, P_hat_f


def solve_adversarial_subproblem(
    w_solution: dict[tuple[int, int, int], float],
    machine_k: int,
    robustness_params: dict[str, Any],
    precomputed: tuple,
    P_hat_f: dict[int, dict[int, dict[int, float]]],
) -> dict[str, Any]:
    """Solve the adversarial subproblem for one machine via greedy.

    Args:
        w_solution: Solution values for w[t, g, k].
        machine_k: Machine index (1-indexed).
        robustness_params: Robustness parameters.
        precomputed: Precomputed data tuple.
        P_hat_f: Per-machine mapping: P_hat_f[k][g][t] -> uncertainty magnitude.

    Returns:
        Dict with keys: worst_case_delay, scenario, lambda_star, mu_star, Gamma_k.
    """
    logger.debug("Solving adversarial subproblem for machine %d", machine_k)
    F, _, Tf, _, _, _, _, _, _, _, _ = precomputed

    # Get per-machine alpha
    raw_alpha = robustness_params["robustness_proportion"]
    if isinstance(raw_alpha, list):
        alpha_k = raw_alpha[machine_k - 1]
    else:
        alpha_k = float(raw_alpha)

    P_hat_k = P_hat_f[machine_k]

    num_batches_on_machine = sum(
        w_solution.get((t, g, machine_k), 0) for g in F for t in Tf[g]
    )
    Gamma_k = int(math.ceil(alpha_k * num_batches_on_machine))

    items = sorted(
        [
            (P_hat_k[g][t], t, g, w_solution.get((t, g, machine_k), 0))
            for g in F
            for t in Tf[g]
            if w_solution.get((t, g, machine_k), 0) > 0
        ],
        reverse=True,
    )

    scenario: dict[tuple[int, int], float] = {}
    remaining_budget = Gamma_k
    worst_case_delay = 0.0
    lambda_star = 0.0

    for p_hat_val, t, g, cap in items:
        if remaining_budget <= 0:
            lambda_star = p_hat_val
            break
        assign = min(cap, remaining_budget)
        scenario[(t, g)] = assign
        worst_case_delay += p_hat_val * assign
        remaining_budget -= assign
        if remaining_budget == 0 and assign < cap:
            lambda_star = p_hat_val
            break

    mu_star = {
        (t, g): max(0.0, P_hat_k[g][t] - lambda_star) for g in F for t in Tf[g]
    }

    return {
        "worst_case_delay": worst_case_delay,
        "scenario": scenario,
        "lambda_star": lambda_star,
        "mu_star": mu_star,
        "Gamma_k": Gamma_k,
    }


def solve_robust(
    jobs: dict[int, tuple[int, int, int]],
    machine_capacity: int,
    num_machines: int,
    robustness_params: dict[str, Any],
    results_data: dict[str, Any],
    time_limit: int | float = 1800,
    mip_gap_abs: float = 0.99,
) -> tuple[str, float | None, dict | None]:
    """Solve the robust model via a single MIP with exact dual embedding.

    Args:
        jobs: Mapping job_id -> (processing_time, size, family).
        machine_capacity: Batch capacity B.
        num_machines: Number of parallel machines.
        robustness_params: Robustness parameters dict.
        results_data: Mutable dict that gets populated with runtime metrics.
        time_limit: Total time limit in seconds.
        mip_gap_abs: Absolute MIP gap tolerance.

    Returns:
        (status_str, makespan, schedule).
    """
    logger.info(
        "Starting robust solve: alpha=%s, p_hat=%s",
        robustness_params["robustness_proportion"],
        robustness_params["p_hat_value"],
    )
    start_time = time.time()

    data = precompute_data(jobs, machine_capacity)
    F, _, Tf, Pf, _, _, _, _, _, _, _ = data

    results_data["rob_runtime_master"] = 0
    results_data["rob_mip_gap"] = None

    master_model, w_vars, eta_vars, Gamma_vars, C_max_var, P_hat_f = build_master_model(
        jobs, machine_capacity, num_machines, robustness_params, data, mip_gap_abs
    )
    master_model.setParam("TimeLimit", time_limit - (time.time() - start_time))

    m_start = time.time()
    master_model.optimize()
    results_data["rob_runtime_master"] = time.time() - m_start
    results_data["rob_total_runtime"] = time.time() - start_time

    if master_model.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        return "Infeasible", None, None
    if master_model.SolCount == 0:
        return "Time Limit", None, None

    ub = master_model.ObjVal
    lb = master_model.ObjBound
    results_data["rob_mip_gap"] = round((ub - lb) / ub, 6) if ub > 0 else 0.0

    w_sol = {k: int(round(v.X)) for k, v in w_vars.items()}
    schedule = reconstruct_schedule(
        jobs, machine_capacity, num_machines, Pf, Tf, F, w_sol
    )

    if master_model.Status == GRB.OPTIMAL:
        return "Optimal", ub, schedule
    return "Time Limit", ub, schedule
