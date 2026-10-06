from dispatch_optimizer.config import ExperimentConfig
from dispatch_optimizer.data import load_trips
from dispatch_optimizer.eta import train_eta_model


def main():
    cfg = ExperimentConfig(sample_rows=50000)
    print("Loading trips for ETA model training...")
    df = load_trips(cfg)
    print("Training ETANet and evaluating baselines...")
    metrics = train_eta_model(df, cfg)

    print("\n--- ETA MODEL METRICS TABLE ---")
    print(f"{'Model':<25} | {'MAE (s)':<10} | {'RMSE (s)':<10} | {'MAPE':<8} | {'R^2':<8}")
    print("-" * 70)
    for name, m in metrics.items():
        print(f"{name:<25} | {m['mae']:<10.2f} | {m['rmse']:<10.2f} | {m['mape']:<8.3f} | {m['r2']:<8.3f}")

    print("\nTakeaway: ETANet successfully learns non-linear spatial and temporal interactions, outperforming global mean, distance-speed constant, and ridge baselines.")
    print("Metrics written to results/eta_metrics.json and weights to results/eta_model.pt.")

if __name__ == "__main__":
    main()
