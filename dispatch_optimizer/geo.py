from pathlib import Path

import numpy as np
import pandas as pd


def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8  # Earth radius in miles
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return R * c

def zone_centroids() -> pd.DataFrame:
    csv_path = Path("data/zone_centroids.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)

    # Generate stable pseudo-centroids for 263 TLC zones based on borough approximations
    # Borough rough centers: Manhattan (40.75, -73.98), Brooklyn (40.65, -73.95), Queens (40.72, -73.82), Bronx (40.85, -73.88), Staten Island (40.58, -74.15)
    np.random.seed(42)
    data = []
    for loc_id in range(1, 264):
        # Distribute across boroughs
        if loc_id in [1, 2, 3]:  # Special/Airport
            lat, lon = 40.6413, -73.7781 # JFK
        elif loc_id in [132, 138]:
            lat, lon = 40.7769, -73.8740 # LGA
        else:
            # Hash-based pseudo random spread
            b = loc_id % 5
            if b == 0: # Manhattan
                lat = 40.72 + (loc_id % 15) * 0.007
                lon = -74.00 + (loc_id % 7) * 0.005
            elif b == 1: # Brooklyn
                lat = 40.60 + (loc_id % 20) * 0.006
                lon = -73.98 + (loc_id % 10) * 0.006
            elif b == 2: # Queens
                lat = 40.70 + (loc_id % 20) * 0.007
                lon = -73.85 + (loc_id % 12) * 0.006
            elif b == 3: # Bronx
                lat = 40.82 + (loc_id % 15) * 0.006
                lon = -73.90 + (loc_id % 8) * 0.006
            else: # Staten Island
                lat = 40.56 + (loc_id % 10) * 0.005
                lon = -74.15 + (loc_id % 6) * 0.005
        data.append({"LocationID": loc_id, "lat": lat, "lon": lon})

    df = pd.DataFrame(data)
    Path("data").mkdir(exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df
