"""Tests for synthetic demand generation, LSTM windowing, and Prophet forecasting."""
from __future__ import annotations

import numpy as np
import pytest

from src.forecasting.data_prep import generate_synthetic_demand, make_lstm_windows, train_val_test_split_series


class TestSyntheticDemand:
    def test_generates_expected_shape(self):
        df = generate_synthetic_demand(entity_ids=["A", "B"], n_days=100, seed=1)
        assert len(df) == 200
        assert set(df["entity_id"].unique()) == {"A", "B"}

    def test_demand_is_non_negative(self):
        df = generate_synthetic_demand(entity_ids=["A"], n_days=200, seed=1)
        assert (df["y"] >= 0).all()

    def test_reproducible_given_seed(self):
        df1 = generate_synthetic_demand(entity_ids=["A"], n_days=50, seed=99)
        df2 = generate_synthetic_demand(entity_ids=["A"], n_days=50, seed=99)
        np.testing.assert_array_almost_equal(df1["y"].to_numpy(), df2["y"].to_numpy())


class TestWindowing:
    def test_window_shapes(self):
        df = generate_synthetic_demand(entity_ids=["A"], n_days=200, seed=1)
        windows = make_lstm_windows(df["y"])
        assert windows.X.shape == (200 - 90 - 30 + 1, 90, 1)
        assert windows.y.shape == (200 - 90 - 30 + 1, 30)

    def test_train_val_test_split_sums_to_total(self):
        df = generate_synthetic_demand(entity_ids=["A"], n_days=100, seed=1)
        train, val, test = train_val_test_split_series(df, "A")
        assert len(train) + len(val) + len(test) == 100


class TestLSTM:
    def test_output_has_thirty_day_horizon(self):
        torch = pytest.importorskip("torch")
        from src.forecasting.lstm_model import DemandLSTM

        model = DemandLSTM(hidden_size=8, num_layers=1)
        predictions = model(torch.zeros((2, 90, 1)))
        assert predictions.shape == (2, 30)

    def test_training_early_stops_and_evaluates_best_checkpoint(self, tmp_path, monkeypatch):
        from src.forecasting import train

        monkeypatch.setattr(train, "LSTM_DIR", tmp_path)
        df = generate_synthetic_demand(entity_ids=["A"], n_days=300, seed=7)
        metrics = train.train_lstm_for_entity(df, "A", epochs=2, patience=1)
        assert {"test_mae", "test_rmse", "test_mape"}.issubset(metrics)
        assert (tmp_path / "lstm_A.pt").exists()


class TestProphet:
    def test_prophet_fits_and_predicts(self):
        pytest.importorskip("prophet")
        from src.forecasting.prophet_model import ProphetForecaster

        df = generate_synthetic_demand(entity_ids=["A"], n_days=120, seed=1)
        entity_df = df[df["entity_id"] == "A"][["ds", "y"]]
        forecaster = ProphetForecaster()
        forecaster.fit(entity_df)
        forecast = forecaster.predict(periods=7)
        assert len(forecast) == 120 + 7
        assert {"ds", "yhat", "yhat_lower", "yhat_upper"}.issubset(forecast.columns)
