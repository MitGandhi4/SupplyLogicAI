"""
Train both the LSTM and Prophet demand forecasters for every manufacturer
entity and log all runs, hyperparameters, and evaluation metrics (MAE,
RMSE, MAPE) to MLflow.

Run:
    python -m src.forecasting.train
"""
from __future__ import annotations

import json

import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import settings
from src.forecasting.data_prep import load_demand_data, make_lstm_windows, train_val_test_split_series
from src.forecasting.lstm_model import DemandLSTM
from src.forecasting.prophet_model import ProphetForecaster
from src.logging_config import get_logger

logger = get_logger(__name__)

LSTM_DIR = settings.model_dir / "lstm"
LSTM_DIR.mkdir(parents=True, exist_ok=True)


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.clip(np.abs(y_true), 1e-3, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def train_lstm_for_entity(
    df,
    entity_id: str,
    window: int = 90,
    horizon: int = 30,
    epochs: int = 60,
    lr: float = 1e-3,
    patience: int = 8,
) -> dict:
    if patience < 1:
        raise ValueError("patience must be at least 1")
    train_df, val_df, test_df = train_val_test_split_series(df, entity_id)
    train_windows = make_lstm_windows(train_df["y"], window=window, horizon=horizon)
    if not len(train_windows.X):
        raise ValueError(f"Not enough training observations for {window}-day input and {horizon}-day forecast")
    mean, std = train_windows.scaler_mean, train_windows.scaler_std

    def _windows_with_scaler(history, targets, mean, std, window, horizon):
        history_values = history.to_numpy(dtype=np.float32)[-window:]
        target_values = targets.to_numpy(dtype=np.float32)
        scaled_history = (history_values - mean) / std
        scaled_targets = (target_values - mean) / std
        X, y = [], []
        for i in range(len(scaled_targets) - horizon + 1):
            context = np.concatenate((scaled_history, scaled_targets[:i]))
            X.append(context[-window:])
            y.append(scaled_targets[i:i + horizon])
        if not X:
            return None, None
        return (
            np.array(X, dtype=np.float32)[..., np.newaxis],
            np.array(y, dtype=np.float32),
        )

    val_X, val_y = _windows_with_scaler(train_df["y"], val_df["y"], mean, std, window, horizon)
    test_history = pd.concat((train_df["y"], val_df["y"]), ignore_index=True)
    test_X, test_y = _windows_with_scaler(test_history, test_df["y"], mean, std, window, horizon)

    model = DemandLSTM(input_size=1, hidden_size=32, num_layers=2, dropout=0.2, forecast_horizon=horizon)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(torch.tensor(train_windows.X), torch.tensor(train_windows.y))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    checkpoint_path = LSTM_DIR / f"lstm_{entity_id}.pt"
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_ds)

        if val_X is not None and len(val_X) > 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(torch.tensor(val_X))
                val_loss = criterion(val_pred, torch.tensor(val_y)).item()
        else:
            val_loss = epoch_loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1

        if epoch % 20 == 0 or epoch == epochs:
            logger.info("[%s] epoch %03d | train_loss=%.4f val_loss=%.4f", entity_id, epoch, epoch_loss, val_loss)
        if epochs_without_improvement >= patience:
            logger.info("[%s] early stopping at epoch %d after %d unimproved epochs", entity_id, epoch, patience)
            break

    # Test-set evaluation in original (unscaled) units.
    metrics = {"val_loss": best_val_loss}
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    if test_X is not None and len(test_X) > 0:
        model.eval()
        with torch.no_grad():
            test_pred_scaled = model(torch.tensor(test_X)).numpy()
        test_pred = (test_pred_scaled * std + mean).reshape(-1)
        test_true = (test_y * std + mean).reshape(-1)
        metrics.update({
            "test_mae": _mae(test_true, test_pred),
            "test_rmse": _rmse(test_true, test_pred),
            "test_mape": _mape(test_true, test_pred),
        })
    return metrics


def train_prophet_for_entity(df, entity_id: str, test_frac: float = 0.15) -> dict:
    train_df, val_df, test_df = train_val_test_split_series(df, entity_id, val_frac=0.0, test_frac=test_frac)
    forecaster = ProphetForecaster()
    forecaster.fit(train_df)
    forecast = forecaster.predict(periods=len(test_df))
    merged = test_df.merge(forecast[["ds", "yhat"]], on="ds", how="inner")
    if merged.empty:
        return {"test_mae": float("nan"), "test_rmse": float("nan"), "test_mape": float("nan")}
    return {
        "test_mae": _mae(merged["y"].to_numpy(), merged["yhat"].to_numpy()),
        "test_rmse": _rmse(merged["y"].to_numpy(), merged["yhat"].to_numpy()),
        "test_mape": _mape(merged["y"].to_numpy(), merged["yhat"].to_numpy()),
    }


def train_all_forecasters(entity_ids: list[str] | None = None) -> dict:
    df = load_demand_data()
    entity_ids = entity_ids or sorted(df["entity_id"].unique().tolist())

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    all_results = {}
    for entity_id in entity_ids:
        with mlflow.start_run(run_name=f"forecast_{entity_id}"):
            mlflow.log_param("entity_id", entity_id)
            mlflow.log_param("model_family", "lstm+prophet")

            lstm_metrics = train_lstm_for_entity(df, entity_id)
            mlflow.log_metrics({f"lstm_{k}": v for k, v in lstm_metrics.items() if v == v})

            prophet_metrics = train_prophet_for_entity(df, entity_id)
            mlflow.log_metrics({f"prophet_{k}": v for k, v in prophet_metrics.items() if v == v})

            lstm_mae = lstm_metrics.get("test_mae")
            prophet_mae = prophet_metrics.get("test_mae")
            comparison = {"preferred_model": None, "test_mae_difference": None}
            if lstm_mae is not None and prophet_mae is not None and np.isfinite([lstm_mae, prophet_mae]).all():
                comparison = {
                    "preferred_model": "lstm" if lstm_mae <= prophet_mae else "prophet",
                    "test_mae_difference": abs(lstm_mae - prophet_mae),
                }
                mlflow.log_param("preferred_model", comparison["preferred_model"])

            all_results[entity_id] = {
                "lstm": lstm_metrics,
                "prophet": prophet_metrics,
                "comparison": comparison,
            }
            logger.info("[%s] lstm=%s prophet=%s", entity_id, lstm_metrics, prophet_metrics)

    results_path = settings.model_dir / "forecast_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    logger.info("Forecast training complete for %d entities. Results at %s", len(entity_ids), results_path)
    return all_results


if __name__ == "__main__":
    train_all_forecasters()
