import json
import time
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


def run_benchmark():
    cfg = ExperimentConfig(sample_rows=30000)
    df = load_trips(cfg)

    windows = [10, 30, 60, 120, 300]
    bench_results = []

    for w in windows:
        cfg_w = ExperimentConfig(batch_window_seconds=w, n_windows=5, sample_rows=30000)
        batches = build_batches(df, cfg_w)

        solve_times_ms = []
        objectives = []
        waits = []
        deadheads = []
        match_rates = []

        for batch in batches:
            cost = build_cost_matrix(batch, cfg_w)
            mask = batch.feasibility_mask

            start_t = time.time()
            match, obj, _ = solve_assignment_ortools(cost, mask)
            elapsed_ms = (time.time() - start_t) * 1000.0

            eval_res = evaluate_solution(batch, match, cfg_w)

            solve_times_ms.append(elapsed_ms)
            objectives.append(obj)
            waits.append(eval_res['mean_wait_seconds'])
            deadheads.append(eval_res['total_deadhead_miles'])
            match_rates.append(eval_res['match_rate'])

        p50_solve = float(np.percentile(solve_times_ms, 50))
        p95_solve = float(np.percentile(solve_times_ms, 95))

        bench_results.append({
            "batch_window_seconds": w,
            "p50_solve_ms": p50_solve,
            "p95_solve_ms": p95_solve,
            "mean_wait_seconds": float(np.mean(waits)),
            "total_deadhead_miles": float(np.mean(deadheads)),
            "match_rate": float(np.mean(match_rates))
        })

    # Scaling exponent fit (log(solve_ms) vs log(batch_window))
    log_w = np.log([r['batch_window_seconds'] for r in bench_results])
    log_t = np.log([max(1.0, r['p50_solve_ms']) for r in bench_results])
    slope, intercept = np.polyfit(log_w, log_t, 1)

    output = {
        "scaling_exponent_fit_slope": float(slope),
        "scaling_exponent_fit_intercept": float(intercept),
        "benchmark_results": bench_results
    }

    Path("results").mkdir(exist_ok=True)
    with open("results/benchmark.json", "w") as f:
        json.dump(output, f, indent=2)

    # Figures
    Path("figures").mkdir(exist_ok=True)

    windows_arr = [r['batch_window_seconds'] for r in bench_results]
    p50_arr = [r['p50_solve_ms'] for r in bench_results]
    mean_wait_arr = [r['mean_wait_seconds'] for r in bench_results]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(windows_arr, p50_arr, 'o-', color='tab:blue', label='P50 Solve (ms)')
    ax1.set_xlabel("Batch Window (Seconds)")
    ax1.set_ylabel("Solve Time (ms)", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(windows_arr, mean_wait_arr, 's--', color='tab:orange', label='Mean Wait (s)')
    ax2.set_ylabel("Mean Wait Seconds", color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.title("Latency vs Mean Wait across Batch Window Sizes")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("figures/latency_vs_batch_window.png", dpi=300)
    plt.close()

    # Problem size log-log plot
    plt.figure(figsize=(6, 4))
    plt.loglog(windows_arr, p50_arr, 'o-', color='purple', lw=2)
    plt.xlabel("Batch Window (Seconds) [Log Scale]")
    plt.ylabel("P50 Solve Time (ms) [Log Scale]")
    plt.title(f"Problem Scale Latency Scaling (Exponent: {slope:.2f})")
    plt.grid(True, linestyle='--', alpha=0.5, which='both')
    plt.tight_layout()
    plt.savefig("figures/latency_vs_problem_size.png", dpi=300)
    plt.close()

    print("Latency benchmark completed successfully.")

if __name__ == "__main__":
    run_benchmark()
