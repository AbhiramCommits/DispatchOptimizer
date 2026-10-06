import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dispatch_optimizer.config import ExperimentConfig
from dispatch_optimizer.data import load_trips
from dispatch_optimizer.instances import build_batches
from dispatch_optimizer.optimize import (
    build_cost_matrix,
    evaluate_solution,
    solve_assignment_ortools,
    solve_greedy,
)


def main():
    cfg = ExperimentConfig(n_windows=5, sample_rows=20000)
    print("Loading trips for matching optimization...")
    df = load_trips(cfg)
    batches = build_batches(df, cfg)

    results = []
    opt_costs = []
    greedy_costs = []

    for batch in batches:
        cost = build_cost_matrix(batch, cfg)
        mask = batch.feasibility_mask

        # Optimal (Min-Cost Flow)
        opt_match, opt_obj, opt_time = solve_assignment_ortools(cost, mask)
        opt_eval = evaluate_solution(batch, opt_match, cfg)

        # Greedy
        greedy_match, greedy_time = solve_greedy(batch, cost, mask)
        greedy_eval = evaluate_solution(batch, greedy_match, cfg)

        opt_costs.append(opt_eval['total_cost'])
        greedy_costs.append(greedy_eval['total_cost'])

        results.append({
            "batch_id": batch.batch_id,
            "optimal": opt_eval,
            "greedy": greedy_eval,
            "opt_solve_ms": opt_time * 1000.0,
            "greedy_solve_ms": greedy_time * 1000.0
        })

    mean_opt_cost = np.mean(opt_costs)
    mean_greedy_cost = np.mean(greedy_costs)
    optimality_gap = ((mean_greedy_cost - mean_opt_cost) / (mean_opt_cost + 1e-8)) * 100.0

    summary = {
        "n_batches": len(batches),
        "mean_optimal_cost": float(mean_opt_cost),
        "mean_greedy_cost": float(mean_greedy_cost),
        "optimality_gap_percent": float(optimality_gap),
        "batch_results": results
    }

    Path("results").mkdir(exist_ok=True)
    with open("results/matching_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Figures
    Path("figures").mkdir(exist_ok=True)
    methods = ['Optimal (Min-Cost)', 'Greedy']
    mean_waits = [
        np.mean([r['optimal']['mean_wait_seconds'] for r in results]),
        np.mean([r['greedy']['mean_wait_seconds'] for r in results])
    ]
    mean_dh = [
        np.mean([r['optimal']['total_deadhead_miles'] for r in results]),
        np.mean([r['greedy']['total_deadhead_miles'] for r in results])
    ]

    plt.figure(figsize=(6, 4))
    plt.bar(methods, mean_waits, color=['forestgreen', 'darkorange'], alpha=0.8)
    plt.ylabel("Mean Wait Seconds")
    plt.title("Wait Time Comparison: Optimal vs Greedy")
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig("figures/wait_vs_method.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(methods, mean_dh, color=['forestgreen', 'darkorange'], alpha=0.8)
    plt.ylabel("Total Deadhead Miles")
    plt.title("Deadhead Miles Comparison: Optimal vs Greedy")
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig("figures/deadhead_vs_method.png", dpi=300)
    plt.close()

    print("\n--- MATCHING RESULTS TABLE ---")
    print("Method     | Mean Wait (s) | P90 Wait (s) | Deadhead (mi) | Match Rate | Optimality Gap")
    print("-" * 80)
    opt_w = np.mean([r['optimal']['mean_wait_seconds'] for r in results])
    opt_p90 = np.mean([r['optimal']['p90_wait_seconds'] for r in results])
    opt_d = np.mean([r['optimal']['total_deadhead_miles'] for r in results])
    opt_m = np.mean([r['optimal']['match_rate'] for r in results])
    print(f"Optimal    | {opt_w:<13.1f} | {opt_p90:<12.1f} | {opt_d:<13.1f} | {opt_m:<10.2f} | 0.0% (Baseline)")

    gr_w = np.mean([r['greedy']['mean_wait_seconds'] for r in results])
    gr_p90 = np.mean([r['greedy']['p90_wait_seconds'] for r in results])
    gr_d = np.mean([r['greedy']['total_deadhead_miles'] for r in results])
    gr_m = np.mean([r['greedy']['match_rate'] for r in results])
    print(f"Greedy     | {gr_w:<13.1f} | {gr_p90:<12.1f} | {gr_d:<13.1f} | {gr_m:<10.2f} | +{optimality_gap:.1f}%")
    print(f"\nTakeaway: OR-Tools min-cost assignment achieves lower total cost and wait times than greedy matching, with an optimality gap of {optimality_gap:.1f}%.")

if __name__ == "__main__":
    main()
