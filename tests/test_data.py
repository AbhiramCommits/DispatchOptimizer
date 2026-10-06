from dispatch_optimizer.config import ExperimentConfig
from dispatch_optimizer.data import synthetic_fallback


def test_synthetic_fallback():
    cfg = ExperimentConfig(sample_rows=500)
    df = synthetic_fallback(cfg)
    assert len(df) == 500
    assert 'request_ts' in df.columns
    assert 'pu_lat' in df.columns
    assert df['data_source'].iloc[0] == 'synthetic_fallback'
