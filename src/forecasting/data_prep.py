"""
Synthetic historical demand/order data generation and windowing utilities
for the LSTM and Prophet forecasting models.

Real order-history data with the right shape (daily demand per
manufacturer, multi-year) is not reliably available from the public Kaggle
supply-chain datasets, so SupplyMind generates a realistic synthetic demand
series per manufacturer node: trend + weekly/annual seasonality + random
noise + occasional disruption shocks (correlated with the causal model's
disruption events). This keeps the forecasting pipeline fully runnable
without blocking on dataset access.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)

DEMAND_CSV_PATH = settings.processed_data_dir / "demand_history.csv"


def generate_synthetic_demand(
    entity_ids: list[str] | None = None,
    n_days: int = 730,
    start_date: str = "2024-01-01",
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate daily demand series for each entity (typically manufacturers).

    Returns a long-format DataFrame with columns: entity_id, ds, y,
    is_disruption (1 if a synthetic disruption shock hit that day).
    """
    seed = seed if seed is not None else settings.random_seed
    entity_ids = entity_ids or [f"MFG_{i:03d}" for i in range(settings.n_manufacturers)]
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start=start_date, periods=n_days, freq="D")
    frames = []

    for idx, entity_id in enumerate(entity_ids):
        base_level = rng.uniform(500, 2000)
        trend_slope = rng.uniform(-0.3, 1.0)
        weekly_amp = rng.uniform(0.05, 0.2) * base_level
        annual_amp = rng.uniform(0.1, 0.3) * base_level
        noise_std = 0.05 * base_level

        t = np.arange(n_days)
        trend = base_level + trend_slope * t
        weekly = weekly_amp * np.sin(2 * np.pi * t / 7)
        annual = annual_amp * np.sin(2 * np.pi * t / 365.25)
        noise = rng.normal(0, noise_std, size=n_days)

        y = trend + weekly + annual + noise

        # Random disruption shocks: sharp demand drop for a short window.
        # Guard against short series (small n_days) where a 30-day margin on
        # both sides would leave no valid shock-start range.
        is_disruption = np.zeros(n_days, dtype=int)
        margin = min(30, max(1, n_days // 4))
        if n_days - 2 * margin > 1:
            n_shocks = rng.integers(2, 6)
            for _ in range(n_shocks):
                shock_start = rng.integers(margin, n_days - margin)
                shock_len = rng.integers(5, max(6, min(21, n_days - shock_start)))
                shock_magnitude = rng.uniform(0.2, 0.6)
                end = min(shock_start + shock_len, n_days)
                y[shock_start:end] *= (1 - shock_magnitude)
                is_disruption[shock_start:end] = 1

        y = np.clip(y, a_min=0, a_max=None)

        frames.append(pd.DataFrame({
            "entity_id": entity_id,
            "ds": dates,
            "y": y,
            "is_disruption": is_disruption,
        }))

    df = pd.concat(frames, ignore_index=True)
    logger.info("Generated synthetic demand data: %d entities x %d days = %d rows", len(entity_ids), n_days, len(df))
    return df


def save_demand_data(df: pd.DataFrame, path=DEMAND_CSV_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved demand data to %s", path)


def load_demand_data(path=DEMAND_CSV_PATH) -> pd.DataFrame:
    if not path.exists():
        logger.info("No cached demand data at %s; generating fresh synthetic data", path)
        df = generate_synthetic_demand()
        save_demand_data(df)
        return df
    return pd.read_csv(path, parse_dates=["ds"])


@dataclass
class WindowedDataset:
    X: np.ndarray  # (n_samples, window, n_features)
    y: np.ndarray  # (n_samples, horizon)
    scaler_mean: float
    scaler_std: float


def make_lstm_windows(series: pd.Series, window: int = 90, horizon: int = 30) -> WindowedDataset:
    """Slice demand into standardized 90-day history -> 30-day forecast samples."""
    values = series.to_numpy(dtype=np.float32)
    mean, std = float(values.mean()), float(values.std() + 1e-8)
    scaled = (values - mean) / std

    X, y = [], []
    for i in range(len(scaled) - window - horizon + 1):
        X.append(scaled[i:i + window])
        y.append(scaled[i + window:i + window + horizon])

    X_arr = np.array(X, dtype=np.float32)[..., np.newaxis]  # (n, window, 1)
    y_arr = np.array(y, dtype=np.float32).reshape(-1, horizon)
    return WindowedDataset(X=X_arr, y=y_arr, scaler_mean=mean, scaler_std=std)


def train_val_test_split_series(df: pd.DataFrame, entity_id: str, val_frac: float = 0.15, test_frac: float = 0.15):
    entity_df = df[df["entity_id"] == entity_id].sort_values("ds").reset_index(drop=True)
    n = len(entity_df)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    train = entity_df.iloc[:n_train]
    val = entity_df.iloc[n_train:n_train + n_val]
    test = entity_df.iloc[n_train + n_val:]
    return train, val, test
