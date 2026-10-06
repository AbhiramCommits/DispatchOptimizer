import time

import numpy as np
from ortools.graph.python import min_cost_flow
from ortools.linear_solver import pywraplp
from scipy.optimize import linprog

from .config import ExperimentConfig
from .eta import predict_pickup_eta
from .geo import haversine_miles
from .instances import DispatchBatch


def build_cost_matrix(batch: DispatchBatch, cfg: ExperimentConfig, lambda_val: float = 0.0) -> np.ndarray:
    n_drv = len(batch.drivers)
    n_req = len(batch.requests)

    cost_matrix = np.zeros((n_drv, n_req))

    for i in range(n_drv):
        d_pos = (batch.drivers.iloc[i]['lat'], batch.drivers.iloc[i]['lon'])
        for j in range(n_req):
            r_pos = (batch.requests.iloc[j]['lat'], batch.requests.iloc[j]['lon'])

            if not batch.feasibility_mask[i, j]:
                cost_matrix[i, j] = 1e9
                continue

            eta_sec = predict_pickup_eta(d_pos, r_pos, batch.start_ts)
            wait_min = eta_sec / 60.0
            deadhead_miles = haversine_miles(d_pos[0], d_pos[1], r_pos[0], r_pos[1])

            # Objective: wait_cost + deadhead_cost - lambda_val * scarcity_value
            # Scarcity value: reserve value or zone scarcity
            scarcity_val = 0.0

            cost = (cfg.wait_cost_per_minute * wait_min +
                    cfg.deadhead_cost_per_mile * deadhead_miles -
                    lambda_val * scarcity_val)
            cost_matrix[i, j] = max(0.0, cost)

    return cost_matrix

def solve_assignment_ortools(cost_matrix: np.ndarray, mask: np.ndarray) -> tuple:
    # Uses OR-Tools min cost flow
    start_t = time.time()
    n_drv, n_req = cost_matrix.shape

    smcf = min_cost_flow.SimpleMinCostFlow()

    # Node indexing:
    # Source = 0
    # Drivers = 1 to n_drv
    # Requests = n_drv + 1 to n_drv + n_req
    # Sink = n_drv + n_req + 1

    source = 0
    sink = n_drv + n_req + 1

    # Scale costs to integers for min cost flow
    scale = 1000.0

    for i in range(n_drv):
        smcf.add_arc_with_capacity_and_unit_cost(source, i + 1, 1, 0)

    for j in range(n_req):
        smcf.add_arc_with_capacity_and_unit_cost(n_drv + 1 + j, sink, 1, 0)
        # Unmatched request arc from source to request with unmatched penalty cost
        smcf.add_arc_with_capacity_and_unit_cost(source, n_drv + 1 + j, 1, int(100.0 * scale))

    for i in range(n_drv):
        for j in range(n_req):
            if mask[i, j] and cost_matrix[i, j] < 1e8:
                c = int(cost_matrix[i, j] * scale)
                smcf.add_arc_with_capacity_and_unit_cost(i + 1, n_drv + 1 + j, 1, c)

    smcf.set_node_supply(source, n_req)
    smcf.set_node_supply(sink, -n_req)

    status = smcf.solve()
    solve_sec = time.time() - start_t

    matching = {}
    total_obj = 0.0
    if status == smcf.OPTIMAL:
        for i in range(smcf.num_arcs()):
            if smcf.flow(i) > 0:
                tail = smcf.tail(i)
                head = smcf.head(i)
                if 1 <= tail <= n_drv and n_drv + 1 <= head <= n_drv + n_req:
                    drv_idx = tail - 1
                    req_idx = head - (n_drv + 1)
                    matching[drv_idx] = req_idx
                    total_obj += smcf.unit_cost(i) / scale
                elif tail == source and n_drv + 1 <= head <= n_drv + n_req:
                    # Unmatched request
                    total_obj += smcf.unit_cost(i) / scale

    return matching, total_obj, solve_sec

def solve_assignment_mip(cost_matrix: np.ndarray, mask: np.ndarray, unmatched_penalty: float = 100.0) -> tuple:
    start_t = time.time()
    n_drv, n_req = cost_matrix.shape

    solver = pywraplp.Solver.CreateSolver('CBC')
    if not solver:
        return {}, 0.0, 0.0

    x = {}
    for i in range(n_drv):
        for j in range(n_req):
            if mask[i, j] and cost_matrix[i, j] < 1e8:
                x[(i, j)] = solver.BoolVar(f'x_{i}_{j}')

    unmatched = {}
    for j in range(n_req):
        unmatched[j] = solver.BoolVar(f'unmatched_{j}')

    # Each driver assigned at most once
    for i in range(n_drv):
        constraint = solver.Constraint(0, 1)
        for j in range(n_req):
            if (i, j) in x:
                constraint.SetCoefficient(x[(i, j)], 1)

    # Each request assigned exactly once or unmatched
    for j in range(n_req):
        constraint = solver.Constraint(1, 1)
        for i in range(n_drv):
            if (i, j) in x:
                constraint.SetCoefficient(x[(i, j)], 1)
        constraint.SetCoefficient(unmatched[j], 1)

    # Objective
    objective = solver.Objective()
    for (i, j), var in x.items():
        objective.SetCoefficient(var, float(cost_matrix[i, j]))
    for j, var in unmatched.items():
        objective.SetCoefficient(var, float(unmatched_penalty))
    objective.SetMinimization()

    status = solver.Solve()
    solve_sec = time.time() - start_t

    matching = {}
    total_obj = solver.Objective().Value() if status == pywraplp.Solver.OPTIMAL else 0.0
    if status == pywraplp.Solver.OPTIMAL:
        for (i, j), var in x.items():
            if var.solution_value() > 0.5:
                matching[i] = j

    return matching, total_obj, solve_sec

def solve_lp_relaxation(cost_matrix: np.ndarray, mask: np.ndarray, unmatched_penalty: float = 100.0) -> tuple:
    n_drv, n_req = cost_matrix.shape
    # Using scipy.optimize.linprog
    # Variables: x_{i,j} for valid pairs, u_j for unmatched
    valid_pairs = [(i, j) for i in range(n_drv) for j in range(n_req) if mask[i, j] and cost_matrix[i, j] < 1e8]
    n_vars = len(valid_pairs) + n_req

    c = np.zeros(n_vars)
    for idx, (i, j) in enumerate(valid_pairs):
        c[idx] = cost_matrix[i, j]
    for j in range(n_req):
        c[len(valid_pairs) + j] = unmatched_penalty

    # Constraints:
    # 1. Driver capacity <= 1
    A_eq = []
    b_eq = []

    # 2. Request fulfillment = 1
    for j in range(n_req):
        row = np.zeros(n_vars)
        for idx, (di, rj) in enumerate(valid_pairs):
            if rj == j:
                row[idx] = 1.0
        row[len(valid_pairs) + j] = 1.0
        A_eq.append(row)
        b_eq.append(1.0)

    A_ub = []
    b_ub = []
    for i in range(n_drv):
        row = np.zeros(n_vars)
        for idx, (di, rj) in enumerate(valid_pairs):
            if di == i:
                row[idx] = 1.0
        A_ub.append(row)
        b_ub.append(1.0)

    res = linprog(c, A_ub=np.array(A_ub) if A_ub else None, b_ub=np.array(b_ub) if b_ub else None,
                  A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=(0, 1), method='highs')

    return res.fun if res.success else 0.0, res.slack if res.success else None

def solve_greedy(batch: DispatchBatch, cost_matrix: np.ndarray, mask: np.ndarray) -> tuple:
    start_t = time.time()
    n_drv, n_req = cost_matrix.shape
    matched_drivers = set()
    matched_requests = set()
    matching = {}

    # Sort all valid pairs by cost ascending
    pairs = []
    for i in range(n_drv):
        for j in range(n_req):
            if mask[i, j] and cost_matrix[i, j] < 1e8:
                pairs.append((cost_matrix[i, j], i, j))

    pairs.sort(key=lambda x: x[0])

    for cost, i, j in pairs:
        if i not in matched_drivers and j not in matched_requests:
            matched_drivers.add(i)
            matched_requests.add(j)
            matching[i] = j

    solve_sec = time.time() - start_t
    return matching, solve_sec

def evaluate_solution(batch: DispatchBatch, matching: dict, cfg: ExperimentConfig) -> dict:
    total_cost = 0.0
    waits = []
    deadheads = []

    matched_reqs = set(matching.values())
    unmatched_count = len(batch.requests) - len(matched_reqs)

    for drv_idx, req_idx in matching.items():
        d_pos = (batch.drivers.iloc[drv_idx]['lat'], batch.drivers.iloc[drv_idx]['lon'])
        r_pos = (batch.requests.iloc[req_idx]['lat'], batch.requests.iloc[req_idx]['lon'])

        eta_sec = predict_pickup_eta(d_pos, r_pos, batch.start_ts)
        wait_min = eta_sec / 60.0
        dh_miles = haversine_miles(d_pos[0], d_pos[1], r_pos[0], r_pos[1])

        waits.append(eta_sec)
        deadheads.append(dh_miles)
        total_cost += cfg.wait_cost_per_minute * wait_min + cfg.deadhead_cost_per_mile * dh_miles

    total_cost += unmatched_count * cfg.unmatched_penalty

    mean_wait = float(np.mean(waits)) if waits else 0.0
    p90_wait = float(np.percentile(waits, 90)) if waits else 0.0
    total_deadhead = float(np.sum(deadheads)) if deadheads else 0.0
    match_rate = float(len(matched_reqs) / len(batch.requests)) if len(batch.requests) > 0 else 0.0

    return {
        "total_cost": total_cost,
        "mean_wait_seconds": mean_wait,
        "p90_wait_seconds": p90_wait,
        "total_deadhead_miles": total_deadhead,
        "match_rate": match_rate,
        "unmatched_count": unmatched_count
    }
