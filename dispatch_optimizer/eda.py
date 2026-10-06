import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import ExperimentConfig


def run_eda(df: pd.DataFrame, cfg: ExperimentConfig) -> dict:
    Path(cfg.out_dir).mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)

    n_trips = len(df)
    min_date = str(df['request_ts'].min())
    max_date = str(df['request_ts'].max())

    df['hour'] = pd.to_datetime(df['request_ts']).dt.hour
    trips_by_hour = df.groupby('hour').size().to_dict()

    median_miles = float(df['trip_miles'].median())
    p90_miles = float(df['trip_miles'].quantile(0.90))
    median_seconds = float(df['trip_seconds'].median())
    p90_seconds = float(df['trip_seconds'].quantile(0.90))

    # Implied speed mph = (trip_miles / (trip_seconds / 3600))
    df['speed_mph'] = df['trip_miles'] / (df['trip_seconds'] / 3600.0)
    df['speed_mph'] = df['speed_mph'].clip(0, 80)
    speed_by_hour = df.groupby('hour')['speed_mph'].mean().to_dict()

    top_zones = df['pu_zone'].value_counts().head(10).to_dict()
    top_zones_str = {str(k): int(v) for k, v in top_zones.items()}

    # Manhattan core zones approx (e.g. 132, 161, 162, 163, 236, 237, etc.)
    manhattan_core = [4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 74, 75, 79, 87, 88, 90, 100, 107, 113, 114, 120, 125, 137, 140, 141, 142, 143, 144, 148, 151, 152, 153, 158, 161, 162, 163, 164, 170, 186, 209, 211, 229, 230, 231, 232, 233, 234, 236, 237, 238, 239, 243, 244, 246, 249, 261, 262, 263]
    manhattan_share = float(df['pu_zone'].isin(manhattan_core).mean())

    summary = {
        "n_trips": n_trips,
        "date_range": [min_date, max_date],
        "trips_by_hour": trips_by_hour,
        "median_trip_miles": median_miles,
        "p90_trip_miles": p90_miles,
        "median_trip_seconds": median_seconds,
        "p90_trip_seconds": p90_seconds,
        "implied_speed_by_hour": speed_by_hour,
        "top_10_pickup_zones": top_zones_str,
        "manhattan_core_share": manhattan_share
    }

    with open(Path(cfg.out_dir) / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Generate Figures
    # 1. Demand by hour
    plt.figure(figsize=(8, 4))
    hours = sorted(trips_by_hour.keys())
    counts = [trips_by_hour[h] for h in hours]
    plt.bar(hours, counts, color='royalblue', alpha=0.8)
    plt.xlabel("Hour of Day")
    plt.ylabel("Trip Count")
    plt.title("TLC Trip Demand by Hour of Day")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("figures/eda_demand_by_hour.png", dpi=300)
    plt.close()

    # 2. Trip duration hist
    plt.figure(figsize=(8, 4))
    plt.hist(df['trip_seconds'] / 60.0, bins=50, color='forestgreen', alpha=0.8, range=(0, 60))
    plt.xlabel("Trip Duration (Minutes)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Trip Durations")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("figures/eda_trip_duration_hist.png", dpi=300)
    plt.close()

    # 3. Speed by hour
    plt.figure(figsize=(8, 4))
    speeds = [speed_by_hour[h] for h in hours]
    plt.plot(hours, speeds, marker='o', color='darkorange', linewidth=2)
    plt.xlabel("Hour of Day")
    plt.ylabel("Mean Implied Speed (MPH)")
    plt.title("Mean Implied Trip Speed by Hour of Day")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("figures/eda_speed_by_hour.png", dpi=300)
    plt.close()

    # 4. Pickup heat (zone volume scatter)
    plt.figure(figsize=(8, 6))
    zone_counts = df['pu_zone'].value_counts().reset_index()
    zone_counts.columns = ['pu_zone', 'count']
    from .geo import zone_centroids
    centroids = zone_centroids()
    z_df = zone_counts.merge(centroids, left_on='pu_zone', right_on='LocationID', how='inner')

    plt.scatter(z_df['lon'], z_df['lat'], s=z_df['count'] / 20.0, c=z_df['count'], cmap='viridis', alpha=0.7)
    plt.colorbar(label='Trip Count')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Pickup Volume Heatmap by Zone Centroid")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("figures/eda_pickup_heat.png", dpi=300)
    plt.close()

    return summary
