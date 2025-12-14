"""Facebook Prophet wrapper for per-entity demand forecasting."""
from __future__ import annotations

import numpy as np
import pandas as pd
from prophet import Prophet
from pandas.tseries.holiday import USFederalHolidayCalendar

from src.logging_config import get_logger

logger = get_logger(__name__)


class ProphetForecaster:
    """Thin wrapper around Prophet with sane defaults for supply chain demand."""

    def __init__(self, weekly_seasonality: bool = True, yearly_seasonality: bool = True, changepoint_prior_scale: float = 0.1):
        self.model = Prophet(
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            changepoint_prior_scale=changepoint_prior_scale,
            interval_width=0.9,
        )
        self.model.add_regressor("is_holiday")
        self.model.add_regressor("economic_indicator")
        self._fitted = False
        self._start_date: pd.Timestamp | None = None

    def _add_regressors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a US holiday flag and reproducible synthetic economic index."""
        result = df.copy()
        result["ds"] = pd.to_datetime(result["ds"])
        if self._start_date is None:
            self._start_date = result["ds"].min().normalize()
        start = self._start_date
        end = result["ds"].max()
        holidays = USFederalHolidayCalendar().holidays(start=start, end=end)
        result["is_holiday"] = result["ds"].dt.normalize().isin(holidays).astype(float)
        elapsed_days = (result["ds"] - start).dt.days.to_numpy(dtype=float)
        result["economic_indicator"] = 100.0 + 0.015 * elapsed_days + 2.0 * np.sin(
            2.0 * np.pi * elapsed_days / 365.25
        )
        return result

    def fit(self, train_df: pd.DataFrame) -> "ProphetForecaster":
        """train_df must have columns ['ds', 'y']."""
        try:
            self._start_date = pd.to_datetime(train_df["ds"]).min().normalize()
            self.model.fit(self._add_regressors(train_df[["ds", "y"]]))
            self._fitted = True
        except Exception:
            logger.exception("Prophet fit failed")
            raise
        return self

    def predict(self, periods: int, freq: str = "D") -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call .fit() before .predict()")
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(self._add_regressors(future))
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
