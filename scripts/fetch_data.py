from dispatch_optimizer.config import ExperimentConfig
from dispatch_optimizer.data import download_tlc, load_trips


def main():
    print("Fetching TLC trip data...")
    path = download_tlc("2024-01")
    print(f"Downloaded/Verified parquet at: {path} (size: {path.stat().st_size if path.exists() else 0} bytes)")

    cfg = ExperimentConfig(sample_rows=5000)
    df = load_trips(cfg)
    print(f"Loaded DataFrame shape: {df.shape}")
    print("First few rows:")
    print(df.head())

if __name__ == "__main__":
    main()
