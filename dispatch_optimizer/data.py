import urllib.request
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .config import ExperimentConfig

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def download_tlc(month: str = "2024-01") -> Path:
    parquet_path = DATA_DIR / f"yellow_tripdata_{month}.parquet"
    if parquet_path.exists() and parquet_path.stat().st_size > 0:
        return parquet_path

    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"
    print(f"Downloading TLC trip data from {url}...")
    try:
        urllib.request.urlretrieve(url, parquet_path)
    except Exception as e:
        print(f"Download failed: {e}")
    return parquet_path

def synthetic_fallback(cfg: ExperimentConfig) -> pd.DataFrame:
    print("WARNING: Using synthetic fallback for TLC trip data due to download failure or missing file.")
    np.random.seed(cfg.seed)
    n = cfg.sample_rows
    start_time = pd.Timestamp(f"{cfg.data_month}-10 08:00:00")
    timestamps = [start_time + pd.Timedelta(seconds=np.random.randint(0, 7200)) for _ in range(n)]

    # Manhattan-ish coordinates (lat: ~40.7 to 40.8, lon: ~-74.0 to -73.9)
    pu_lat = np.random.uniform(40.70, 40.82, n)
    pu_lon = np.random.uniform(-74.01, -73.93, n)
    do_lat = np.random.uniform(40.70, 40.82, n)
    do_lon = np.random.uniform(-74.01, -73.93, n)

    from .geo import haversine_miles
    miles = haversine_miles(pu_lat, pu_lon, do_lat, do_lon) + 0.5
    seconds = miles * np.random.uniform(120, 250, n) + np.random.normal(300, 50, n)
    seconds = np.clip(seconds, 60, 7200)

    df = pd.DataFrame({
        "request_ts": timestamps,
        "pu_zone": np.random.randint(1, 263, n),
        "do_zone": np.random.randint(1, 263, n),
        "pu_lat": pu_lat,
        "pu_lon": pu_lon,
        "do_lat": do_lat,
        "do_lon": do_lon,
        "trip_miles": miles,
        "trip_seconds": seconds,
        "passenger_count": np.random.choice([1, 2, 3, 4], n),
        "data_source": "synthetic_fallback"
    })
    df = df.sort_values("request_ts").reset_index(drop=True)
    return df

def load_trips(cfg: ExperimentConfig) -> pd.DataFrame:
    parquet_path = DATA_DIR / f"yellow_tripdata_{cfg.data_month}.parquet"
    if not parquet_path.exists() or parquet_path.stat().st_size == 0:
        download_tlc(cfg.data_month)

    if not parquet_path.exists() or parquet_path.stat().st_size == 0:
        df = synthetic_fallback(cfg)
        return df

    try:
        con = duckdb.connect(database=':memory:')
        # Check columns or query safely
        query = f"""
            SELECT
                tpep_pickup_datetime AS request_ts,
                CAST(PULocationID AS INTEGER) AS pu_zone,
                CAST(DOLocationID AS INTEGER) AS do_zone,
                trip_distance AS trip_miles,
                EXTRACT(EPOCH FROM (tpep_dropoff_datetime - tpep_pickup_datetime)) AS trip_seconds,
                COALESCE(passenger_count, 1) AS passenger_count
            FROM read_parquet('{parquet_path}')
            WHERE PULocationID IS NOT NULL
              AND DOLocationID IS NOT NULL
              AND trip_distance > 0 AND trip_distance <= 60
              AND fare_amount > 0
              AND tpep_dropoff_datetime > tpep_pickup_datetime
            LIMIT {cfg.sample_rows}
        """
        df = con.execute(query).fetchdf()
        con.close()

        if len(df) < 1000:
            raise ValueError("Too few rows loaded from parquet")

        from .geo import zone_centroids
        centroids = zone_centroids()

        # Merge centroids for pu and do
        df = df.merge(centroids.rename(columns={'LocationID': 'pu_zone', 'lat': 'pu_lat', 'lon': 'pu_lon'}), on='pu_zone', how='left')
        df = df.merge(centroids.rename(columns={'LocationID': 'do_zone', 'lat': 'do_lat', 'lon': 'do_lon'}), on='do_zone', how='left')

        df = df.dropna(subset=['pu_lat', 'pu_lon', 'do_lat', 'do_lon'])
        df['data_source'] = 'nyc_tlc_parquet'
        df['passenger_count'] = df['passenger_count'].clip(1, 6).fillna(1).astype(int)
        df = df.sort_values('request_ts').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Error loading parquet via DuckDB: {e}")
        return synthetic_fallback(cfg)
