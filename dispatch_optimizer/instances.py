from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .geo import haversine_miles


@dataclass
class DispatchBatch:
    batch_id: int
    start_ts: pd.Timestamp
    requests: pd.DataFrame  # columns: req_id, lat, lon, zone
    drivers: pd.DataFrame   # columns: driver_id, lat, lon, zone
    feasibility_mask: np.ndarray  # shape (n_drivers, n_requests) boolean mask

def build_batches(df: pd.DataFrame, cfg: ExperimentConfig) -> List[DispatchBatch]:
    if df.empty:
        return []

    start_time = df['request_ts'].min()
    window_sec = cfg.batch_window_seconds
    batches = []

    np.random.seed(cfg.seed)

    for i in range(cfg.n_windows):
        w_start = start_time + pd.Timedelta(seconds=i * window_sec)
        w_end = w_start + pd.Timedelta(seconds=window_sec)

        subset = df[(df['request_ts'] >= w_start) & (df['request_ts'] < w_end)]
        if len(subset) < 5:
            # Take a small random sample if window is too sparse
            subset = df.sample(n=min(20, len(df)), random_state=cfg.seed + i)

        requests = pd.DataFrame({
            'req_id': range(len(subset)),
            'lat': subset['pu_lat'].values,
            'lon': subset['pu_lon'].values,
            'zone': subset['pu_zone'].values
        })

        n_req = len(requests)
        n_drv = max(1, int(n_req * cfg.supply_ratio))

        # Generate drivers from recent dropoffs or jittered request origins
        if len(subset) > 0 and 'do_lat' in subset.columns:
            base_lats = subset['do_lat'].values
            base_lons = subset['do_lon'].values
            base_zones = subset['do_zone'].values
            drv_lat = np.random.choice(base_lats, n_drv, replace=True) + np.random.normal(0, 0.005, n_drv)
            drv_lon = np.random.choice(base_lons, n_drv, replace=True) + np.random.normal(0, 0.005, n_drv)
            drv_zone = np.random.choice(base_zones, n_drv, replace=True)
        else:
            drv_lat = requests['lat'].values[np.random.choice(n_req, n_drv, replace=True)] + np.random.normal(0, 0.005, n_drv)
            drv_lon = requests['lon'].values[np.random.choice(n_req, n_drv, replace=True)] + np.random.normal(0, 0.005, n_drv)
            drv_zone = requests['zone'].values[np.random.choice(n_req, n_drv, replace=True)]

        drivers = pd.DataFrame({
            'driver_id': range(n_drv),
            'lat': drv_lat,
            'lon': drv_lon,
            'zone': drv_zone
        })

        # Calculate distance matrix (n_drivers, n_requests)
        # Using broadcasting
        d_lat = drivers['lat'].values[:, None]
        d_lon = drivers['lon'].values[:, None]
        r_lat = requests['lat'].values[None, :]
        r_lon = requests['lon'].values[None, :]

        distances = haversine_miles(d_lat, d_lon, r_lat, r_lon)
        feasibility_mask = distances <= cfg.max_pickup_miles

        batches.append(DispatchBatch(
            batch_id=i,
            start_ts=w_start,
            requests=requests,
            drivers=drivers,
            feasibility_mask=feasibility_mask
        ))

    return batches
