import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dispatch_optimizer.config import ExperimentConfig
from dispatch_optimizer.data import load_trips
from dispatch_optimizer.eta import predict_pickup_eta
from dispatch_optimizer.instances import build_batches
from dispatch_optimizer.optimize import (
    evaluate_solution,
    solve_assignment_ortools,
)


def run_eta_sensitivity():
    cfg = ExperimentConfig(n_windows=5, sample_rows=20000)
    df = load_trips(cfg)
    batches = build_batches(df, cfg)

    sigmas = [0.0, 0.1, 0.2, 0.4, 0.8]
    sensitivity_results = []

    for sigma in sigmas:
        waits = []
        for batch in batches:
            # Add noise to ETA coefficients during optimization
            n_drv = len(batch.drivers)
            n_req = len(batch.requests)
            cost = np.zeros((n_drv, n_req))

            for i in range(n_drv):
                d_pos = (batch.drivers.iloc[i]['lat'], batch.drivers.iloc[i]['lon'])
                for j in range(n_req):
                    r_pos = (batch.requests.iloc[j]['lat'], batch.requests.iloc[j]['lon'])
                    if not batch.feasibility_mask[i, j]:
                        cost[i, j] = 1e9
                        continue
                    true_eta = predict_pickup_eta(d_pos, r_pos, batch.start_ts)
                    # Add noise
                    noisy_eta = true_eta * (1.0 + np.random.normal(0, sigma))
                    wait_min = max(0.5, noisy_eta / 60.0)
                    dh_miles = np.sqrt((d_pos[0]-r_pos[0])**2 + (d_pos[1]-r_pos[1])**2) * 69.0
                    cost[i, j] = cfg.wait_cost_per_minute * wait_min + cfg.deadhead_cost_per_mile * dh_miles

            match, _, _ = solve_assignment_ortools(cost, batch.feasibility_mask)
            eval_res = evaluate_solution(batch, match, cfg)
            waits.append(eval_res['mean_wait_seconds'])

        mean_wait = float(np.mean(waits))
        sensitivity_results.append({
            "sigma": float(sigma),
            "mean_wait_seconds": mean_wait
        })

    # Regret slope calculation
    sigmas_arr = np.array([r['sigma'] for r in sensitivity_results])
    waits_arr = np.array([r['mean_wait_seconds'] for r in sensitivity_results])
    regret_slope, _ = np.polyfit(sigmas_arr, waits_arr, 1)

    output = {
        "regret_slope_seconds_per_sigma": float(regret_slope),
        "sensitivity_results": sensitivity_results
    }

    Path("results").mkdir(exist_ok=True)
    with open("results/eta_sensitivity.json", "w") as f:
        json.dump(output, f, indent=2)

    # Figure
    Path("figures").mkdir(exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(sigmas_arr, waits_arr, 'o-', color='darkred', lw=2)
    plt.xlabel("ETA Noise Multiplier (Sigma)")
    plt.ylabel("Realized Mean Wait Seconds")
    plt.title("Wait Time Degradation vs ETA Model Error (Sigma)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("figures/wait_degradation_vs_eta_error.png", dpi=300)
    plt.close()

    print("ETA error sensitivity experiment completed successfully.")

if __name__ == "__main__":
    run_eta_sensitivity()
