import numpy as np

from dispatch_optimizer.geo import haversine_miles, zone_centroids


def test_haversine_identity():
    # Zero distance for same point
    dist = haversine_miles(40.7128, -74.0060, 40.7128, -74.0060)
    assert dist == 0.0

def test_haversine_symmetry():
    lat1, lon1 = 40.7128, -74.0060
    lat2, lon2 = 40.7589, -73.9851
    d1 = haversine_miles(lat1, lon1, lat2, lon2)
    d2 = haversine_miles(lat2, lon2, lat1, lon1)
    assert np.isclose(d1, d2)

def test_zone_centroids():
    centroids = zone_centroids()
    assert len(centroids) == 263
    assert 'LocationID' in centroids.columns
    assert 'lat' in centroids.columns
    assert 'lon' in centroids.columns
