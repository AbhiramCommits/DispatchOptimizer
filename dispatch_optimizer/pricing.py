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
)


def sweep_lambda(batches, lambda_grid, cfg: ExperimentConfig):
    results = []

    for l_val in lambda_grid:
        batch_metrics = []
        for batch in batches:
            # Objective formulation with scarcity multiplier lambda
            cost = build_cost_matrix(batch, cfg, lambda_val=l_val)
            mask = batch.feasibility_mask

            match, obj, solve_t = solve_assignment_ortools(cost, mask)
            eval_res = evaluate_solution(batch, match, cfg)
            eval_res['solve_seconds'] = solve_t
            eval_res['objective_value'] = obj
            batch_metrics.append(eval_res)

        mean_wait = float(np.mean([b['mean_wait_seconds'] for b in batch_metrics]))
        total_dh = float(np.sum([b['total_deadhead_miles'] for b in batch_metrics]))
        match_rate = float(np.mean([b['match_rate'] for b in batch_metrics]))
        mean_solve = float(np.mean([b['solve_seconds'] for b in batch_metrics]))

        results.append({
            "lambda": float(l_val),
            "mean_wait_seconds": mean_wait,
            "total_deadhead_miles": total_dh,
            "match_rate": match_rate,
            "mean_solve_ms": mean_solve * 1000.0,
            "batch_details": batch_metrics
        })

    # Determine Pareto dominance
    # Non-dominated if no other point has both lower wait and lower deadhead
    for i, p in enumerate(results):
        dominated = False
        for j, q in enumerate(results):
            if i != j:
                if q['mean_wait_seconds'] <= p['mean_wait_seconds'] and q['total_deadhead_miles'] <= p['total_deadhead_miles']:
                    if q['mean_wait_seconds'] < p['mean_wait_seconds'] or q['total_deadhead_miles'] < p['total_deadhead_miles']:
                        dominated = True
                        break
        p['pareto_efficient'] = not dominated

    return results

def main():
    cfg = ExperimentConfig(n_windows=5, sample_rows=20000, lambda_grid=[0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
    print("Loading trips for Lagrangian pricing sweep...")
    df = load_trips(cfg)
    batches = build_batches(df, cfg)

    print(f"Sweeping {len(cfg.lambda_grid)} lambda values across {len(batches)} batches...")
    sweep_results = sweep_lambda(batches, cfg.lambda_grid, cfg)

    Path("results").mkdir(exist_ok=True)
    with open("results/pareto_frontier.json", "w") as f:
        json.dump(sweep_results, f, indent=2)

    # Figures
    Path("figures").mkdir(exist_ok=True)

    lambdas = [r['lambda'] for r in sweep_results]
    waits = [r['mean_wait_seconds'] for r in sweep_results]
    deadheads = [r['total_deadhead_miles'] for r in sweep_results]

    # 1. Pareto Wait vs Deadhead
    plt.figure(figsize=(7, 5))
    plt.plot(deadheads, waits, 'o-', color='navy', lw=2, markersize=8)
    for r in sweep_results:
        plt.annotate(f"λ={r['lambda']}", (r['total_deadhead_miles'], r['mean_wait_seconds']),
                     textcoords="offset points", xytext=(0, 10), ha='center')
    plt.xlabel("Total Deadhead Miles")
    plt.ylabel("Mean Wait Seconds")
    plt.title("Wait vs Deadhead Pareto Frontier (Lagrangian Scarcity Pricing)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("figures/pareto_wait_vs_deadhead.png", dpi=300)
    plt.close()

    # 2. Lambda Sensitivity
    plt.figure(figsize=(7, 4))
    plt.plot(lambdas, waits, marker='s', color='crimson', lw=2)
    plt.xlabel("Lambda (Scarcity Multiplier)")
    plt.ylabel("Mean Wait Seconds")
    plt.title("Sensitivity of Mean Wait Time to Scarcity Multiplier Lambda")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("figures/lambda_sensitivity.png", dpi=300)
    plt.close()

    print("\n--- PARETO FRONTIER SUMMARY ---")
    print(f"{'Lambda':<8} | {'Wait (s)':<12} | {'Deadhead (mi)':<15} | {'Match Rate':<12} | {'Pareto Efficient'}")
    print("-" * 75)
    for r in sweep_results:
        print(f"{r['lambda']:<8.1f} | {r['mean_wait_seconds']:<12.1f} | {r['total_deadhead_miles']:<15.1f} | {r['match_rate']:<12.2f} | {str(r['pareto_efficient']):<5}")

    print("\nTakeaway: Sweeping lambda traces the Pareto frontier between rider wait time and driver deadhead miles under supply scarcity constraints.")

if __name__ == "__main__":
    main()
