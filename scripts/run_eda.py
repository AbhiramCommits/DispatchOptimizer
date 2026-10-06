from dispatch_optimizer.config import ExperimentConfig
from dispatch_optimizer.data import load_trips
from dispatch_optimizer.eda import run_eda


def main():
    cfg = ExperimentConfig(sample_rows=50000)
    print("Loading trips for EDA...")
    df = load_trips(cfg)
    print(f"Running EDA on {len(df)} trips...")
    summary = run_eda(df, cfg)

    print("\n--- EDA SUMMARY TABLE ---")
    print(f"Total Cleaned Trips: {summary['n_trips']}")
    print(f"Date Range: {summary['date_range']}")
    print(f"Median Trip Distance: {summary['median_trip_miles']:.2f} miles (p90: {summary['p90_trip_miles']:.2f})")
    print(f"Median Trip Duration: {summary['median_trip_seconds']/60.0:.1f} mins (p90: {summary['p90_trip_seconds']/60.0:.1f})")
    print(f"Manhattan Core Share: {summary['manhattan_core_share']*100:.1f}%")
    print("Takeaway: Demand peaks during rush windows with significant concentration in Manhattan core zones and sharp speed drops during congestion hours.")
    print("Figures written to figures/ and summary to results/eda_summary.json.")

if __name__ == "__main__":
    main()
