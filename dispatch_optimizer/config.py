from dataclasses import dataclass, field
from typing import List


@dataclass
class ExperimentConfig:
    seed: int = 42
    batch_window_seconds: int = 60
    n_windows: int = 20
    grid_resolution: int = 10
    max_pickup_miles: float = 5.0
    deadhead_cost_per_mile: float = 1.5
    wait_cost_per_minute: float = 0.5
    lambda_grid: List[float] = field(default_factory=lambda: [0.0, 0.2, 0.5, 1.0, 2.0, 5.0])
    data_month: str = "2024-01"
    sample_rows: int = 100000
    out_dir: str = "results"
    supply_ratio: float = 1.2
    unmatched_penalty: float = 100.0
