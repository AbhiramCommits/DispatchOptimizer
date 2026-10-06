from dispatch_optimizer.config import ExperimentConfig
from dispatch_optimizer.data import load_trips
from dispatch_optimizer.instances import build_batches
from dispatch_optimizer.pricing import sweep_lambda


def main():
    cfg = ExperimentConfig(n_windows=5, sample_rows=20000, lambda_grid=[0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
    df = load_trips(cfg)
    batches = build_batches(df, cfg)
    sweep_lambda(batches, cfg.lambda_grid, cfg)
    print("Pareto sweep completed successfully.")

if __name__ == "__main__":
    main()
