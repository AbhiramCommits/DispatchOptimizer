import numpy as np
import pandas as pd
import torch

from dispatch_optimizer.config import ExperimentConfig
from dispatch_optimizer.eta import ETANet, train_eta_model


def test_eta_forward_pass():
    model = ETANet(dense_dim=6)
    dense = torch.randn(10, 6)
    pu = torch.randint(1, 263, (10,))
    do = torch.randint(1, 263, (10,))
    out = model(dense, pu, do)
    assert out.shape == (10,)

def test_eta_training_loss():
    df = pd.DataFrame({
        'request_ts': pd.date_range('2024-01-01', periods=100, freq='min'),
        'pu_zone': np.random.randint(1, 20, 100),
        'do_zone': np.random.randint(1, 20, 100),
        'pu_lat': np.random.uniform(40.7, 40.8, 100),
        'pu_lon': np.random.uniform(-74.0, -73.9, 100),
        'do_lat': np.random.uniform(40.7, 40.8, 100),
        'do_lon': np.random.uniform(-74.0, -73.9, 100),
        'passenger_count': [1]*100,
        'trip_seconds': np.random.uniform(300, 1200, 100)
    })
    cfg = ExperimentConfig()
    metrics = train_eta_model(df, cfg)
    assert 'ETANet' in metrics
    assert 'mae' in metrics['ETANet']
