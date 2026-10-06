import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from .config import ExperimentConfig
from .geo import haversine_miles


class ETANet(nn.Module):
    def __init__(self, n_zones=264, embed_dim=16, dense_dim=10):
        super().__init__()
        self.pu_embed = nn.Embedding(n_zones + 1, embed_dim)
        self.do_embed = nn.Embedding(n_zones + 1, embed_dim)
        in_dim = dense_dim + embed_dim * 2
        self.fc1 = nn.Linear(in_dim, 128)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, dense, pu_zone, do_zone):
        p_emb = self.pu_embed(pu_zone)
        d_emb = self.do_embed(do_zone)
        x = torch.cat([dense, p_emb, d_emb], dim=1)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        out = self.fc3(x)
        return out.squeeze(1)

def extract_features(df: pd.DataFrame):
    lat1, lon1 = df['pu_lat'].values, df['pu_lon'].values
    lat2, lon2 = df['do_lat'].values, df['do_lon'].values

    hav = haversine_miles(lat1, lon1, lat2, lon2)
    dx = np.abs(lat2 - lat1) * 69.0
    dy = np.abs(lon2 - lon1) * 53.0

    ts = pd.to_datetime(df['request_ts'])
    hour = ts.dt.hour.values
    dow = ts.dt.dayofweek.values

    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)

    passengers = df['passenger_count'].values.astype(float)

    dense = np.column_stack([hav, dx, dy, hour_sin, hour_cos, passengers])
    # Normalize dense features
    dense_mean = dense.mean(axis=0)
    dense_std = dense.std(axis=0) + 1e-5
    dense_norm = (dense - dense_mean) / dense_std

    pu_zone = df['pu_zone'].values.astype(int).clip(0, 264)
    do_zone = df['do_zone'].values.astype(int).clip(0, 264)

    targets = df['trip_seconds'].values.astype(float)

    return dense_norm, pu_zone, do_zone, targets

def train_eta_model(df: pd.DataFrame, cfg: ExperimentConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Time-ordered split: train on earlier 70%, val 15%, test 15%
    df = df.sort_values('request_ts').reset_index(drop=True)
    n = len(df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.85)

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_val]
    test_df = df.iloc[n_val:]

    X_tr_dense, X_tr_pu, X_tr_do, y_tr = extract_features(train_df)
    X_va_dense, X_va_pu, X_va_do, y_va = extract_features(val_df)
    X_te_dense, X_te_pu, X_te_do, y_te = extract_features(test_df)

    y_tr_log = np.log(np.clip(y_tr, 30, 86400))
    y_va_log = np.log(np.clip(y_va, 30, 86400))

    model = ETANet(dense_dim=X_tr_dense.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # Training loop with early stopping on validation loss
    best_val_loss = float('inf')
    best_weights = None

    batch_size = 256
    n_batches = len(X_tr_dense) // batch_size

    for epoch in range(15):
        model.train()
        permutation = np.random.permutation(len(X_tr_dense))
        for i in range(0, len(X_tr_dense), batch_size):
            idx = permutation[i:i+batch_size]
            b_dense = torch.tensor(X_tr_dense[idx], dtype=torch.float32)
            b_pu = torch.tensor(X_tr_pu[idx], dtype=torch.long)
            b_do = torch.tensor(X_tr_do[idx], dtype=torch.long)
            b_y = torch.tensor(y_tr_log[idx], dtype=torch.float32)

            optimizer.zero_grad()
            pred = model(b_dense, b_pu, b_do)
            loss = criterion(pred, b_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            va_dense = torch.tensor(X_va_dense, dtype=torch.float32)
            va_pu = torch.tensor(X_va_pu, dtype=torch.long)
            va_do = torch.tensor(X_va_do, dtype=torch.long)
            va_y = torch.tensor(y_va_log, dtype=torch.float32)
            val_pred = model(va_dense, va_pu, va_do)
            val_loss = criterion(val_pred, va_y).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()

    if best_weights:
        model.load_state_dict(best_weights)

    Path("results").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "results/eta_model.pt")

    # Baselines and Evaluation on test set
    model.eval()
    with torch.no_grad():
        te_dense = torch.tensor(X_te_dense, dtype=torch.float32)
        te_pu = torch.tensor(X_te_pu, dtype=torch.long)
        te_do = torch.tensor(X_te_do, dtype=torch.long)
        net_pred_log = model(te_dense, te_pu, te_do).numpy()
        net_pred = np.exp(net_pred_log)

    # Baseline 1: Global mean
    b1_pred = np.full_like(y_te, y_tr.mean())

    # Baseline 2: Distance / median speed constant (median speed ~ 12 mph = 0.2 miles/min = 12 miles/hr)
    hav_te = haversine_miles(test_df['pu_lat'].values, test_df['pu_lon'].values, test_df['do_lat'].values, test_df['do_lon'].values)
    median_speed_mps = 12.0 * 5280.0 / 3600.0
    b2_pred = (hav_te * 5280.0) / median_speed_mps

    # Baseline 3: Ridge regression via lstsq on dense features
    X_tr_design = np.column_stack([np.ones(len(X_tr_dense)), X_tr_dense])
    X_te_design = np.column_stack([np.ones(len(X_te_dense)), X_te_dense])
    weights, _, _, _ = np.linalg.lstsq(X_tr_design, y_tr, rcond=None)
    b3_pred = np.clip(X_te_design @ weights, 30, 86400)

    # Baseline 4: Zone pair historical median
    zone_pair_median = train_df.groupby(['pu_zone', 'do_zone'])['trip_seconds'].median().to_dict()
    global_median = float(np.median(y_tr))
    b4_pred = np.array([zone_pair_median.get((int(r.pu_zone), int(r.do_zone)), global_median) for _, r in test_df.iterrows()])

    def evaluate_metrics(y_true, y_pred):
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
        mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))))
        ss_res = float(np.sum((y_true - y_pred)**2))
        ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
        r2 = float(1.0 - ss_res / (ss_tot + 1e-8))
        return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}

    metrics = {
        "ETANet": evaluate_metrics(y_te, net_pred),
        "GlobalMean": evaluate_metrics(y_te, b1_pred),
        "DistanceSpeedConstant": evaluate_metrics(y_te, b2_pred),
        "RidgeRegression": evaluate_metrics(y_te, b3_pred),
        "ZonePairHistorical": evaluate_metrics(y_te, b4_pred)
    }

    with open("results/eta_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Figures
    Path("figures").mkdir(exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_te[:1000] / 60.0, net_pred[:1000] / 60.0, alpha=0.3, color='purple')
    plt.plot([0, 60], [0, 60], 'k--', lw=2)
    plt.xlabel("Actual Trip Duration (Minutes)")
    plt.ylabel("Predicted ETA (Minutes)")
    plt.title("ETANet: Predicted vs Actual Trip Duration")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("figures/eta_pred_vs_actual.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 4))
    residuals = (y_te - net_pred) / 60.0
    plt.hist(residuals, bins=50, range=(-20, 20), color='teal', alpha=0.8)
    plt.xlabel("Residual (Actual - Predicted) in Minutes")
    plt.ylabel("Frequency")
    plt.title("ETANet Residual Distribution")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("figures/eta_residual_hist.png", dpi=300)
    plt.close()

    return metrics

def predict_pickup_eta(driver_pos: tuple, request_pos: tuple, ts, model=None) -> float:
    # driver_pos: (lat, lon), request_pos: (lat, lon)
    hav = haversine_miles(driver_pos[0], driver_pos[1], request_pos[0], request_pos[1])
    # Assume mean speed ~ 12 mph for direct pickup legs, convert to seconds
    seconds = (hav / 12.0) * 3600.0
    return float(np.clip(seconds, 30.0, 3600.0))

def predict_with_noise(driver_pos: tuple, request_pos: tuple, ts, sigma: float = 0.0) -> float:
    true_eta = predict_pickup_eta(driver_pos, request_pos, ts)
    if sigma <= 0:
        return true_eta
    noise = np.random.normal(0, sigma * true_eta)
    return float(np.clip(true_eta + noise, 10.0, 7200.0))
