"""Pydantic request/response schemas for the FastAPI backend."""
from __future__ import annotations

from pydantic import BaseModel, Field


class NodeOut(BaseModel):
    node_id: str
    node_type: str
    name: str
    country: str
    region: str
    latitude: float
    longitude: float
    risk_score: float
    capacity: float
    reliability: float
    tier: int


class EdgeOut(BaseModel):
    source: str
    target: str
    edge_type: str
    weight: float
    lead_time_days: float
    volume: float
    shipping_cost: float = 0.0
    transit_days: float = 0.0
    dependency_strength: float = 0.0


class GraphOut(BaseModel):
    nodes: list[NodeOut]
    edges: list[EdgeOut]


class CriticalNodeOut(BaseModel):
    node_id: str
    composite_score: float


class DisruptionSimulationOut(BaseModel):
    origin_node: str
    affected_nodes: list[str]
    n_hops: int
    total_capacity_lost: float


class ForecastPoint(BaseModel):
    ds: str
    yhat: float
    yhat_lower: float
    yhat_upper: float


class ForecastOut(BaseModel):
    entity_id: str
    history: list[dict]
    forecast: list[ForecastPoint]


class GNNPredictionOut(BaseModel):
    node_id: str
    critical_probability: float
    predicted_impact: float


class GNNEdgePredictionOut(BaseModel):
    source: str
    target: str
    predicted_impact: float


class CausalEffectOut(BaseModel):
    ate: float
    confidence_interval: list[float] | None
    placebo_refutation_passed: bool
    random_common_cause_refutation_passed: bool
    summary: str


class AlertRequest(BaseModel):
    news_query: str = Field(default="supply chain disruption")


class ErrorOut(BaseModel):
    detail: str
