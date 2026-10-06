import numpy as np
import pandas as pd

from dispatch_optimizer.instances import DispatchBatch
from dispatch_optimizer.optimize import (
    solve_assignment_mip,
    solve_assignment_ortools,
    solve_greedy,
)


def test_matching_opt_and_greedy():
    cost = np.array([
        [10.0, 20.0, 30.0],
        [15.0, 10.0, 25.0],
        [30.0, 25.0, 10.0]
    ])
    mask = np.ones((3, 3), dtype=bool)

    match_opt, obj_opt, _ = solve_assignment_ortools(cost, mask)
    match_greedy, _ = solve_greedy(DispatchBatch(0, pd.Timestamp('2024-01-01'), pd.DataFrame({'lat': [0,0,0], 'lon': [0,0,0], 'zone': [1,1,1]}), pd.DataFrame({'lat': [0,0,0], 'lon': [0,0,0], 'zone': [1,1,1]}), mask), cost, mask)

    assert len(match_opt) == 3
    assert obj_opt > 0

def test_mip_equals_flow():
    cost = np.array([
        [12.0, 45.0],
        [30.0, 15.0]
    ])
    mask = np.ones((2, 2), dtype=bool)
    match_flow, obj_flow, _ = solve_assignment_ortools(cost, mask)
    match_mip, obj_mip, _ = solve_assignment_mip(cost, mask)

    assert np.isclose(obj_flow, obj_mip, atol=1e-3)
