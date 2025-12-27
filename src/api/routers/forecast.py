"""Demand forecasting endpoints (Prophet, fit on-the-fly per request)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import ForecastOut, ForecastPoint
from src.forecasting.data_prep import load_demand_data
from src.forecasting.prophet_model import ProphetForecaster
from src.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.get("/{entity_id}", response_model=ForecastOut)
def get_forecast(entity_id: str, periods: int = Query(default=30, ge=1, le=180)) -> ForecastOut:
    df = load_demand_data()
    entity_df = df[df["entity_id"] == entity_id].sort_values("ds")
    if entity_df.empty:
        raise HTTPException(status_code=404, detail=f"No demand history found for entity '{entity_id}'")

    try:
        forecaster = ProphetForecaster()
        forecaster.fit(entity_df[["ds", "y"]])
        forecast_df = forecaster.predict(periods=periods)
    except Exception as exc:
        logger.exception("Forecast generation failed for entity %s", entity_id)
        raise HTTPException(status_code=500, detail=f"Forecast failed: {exc}") from exc

    history = entity_df[["ds", "y"]].tail(90).assign(ds=lambda d: d["ds"].dt.strftime("%Y-%m-%d")).to_dict(orient="records")
    forecast_points = [
        ForecastPoint(ds=row.ds.strftime("%Y-%m-%d"), yhat=row.yhat, yhat_lower=row.yhat_lower, yhat_upper=row.yhat_upper)
        for row in forecast_df.tail(periods).itertuples()
    ]
    return ForecastOut(entity_id=entity_id, history=history, forecast=forecast_points)
