"""Pydantic schema for structured disruption alerts issued by the agent."""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Evidence(BaseModel):
    text: str
    source: str
    score: float = Field(..., ge=-1.0, le=1.0)


class DisruptionAlert(BaseModel):
    alert_id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    severity: Severity
    origin_node: str
    affected_nodes: list[str]
    predicted_capacity_loss_fraction: float = Field(..., ge=0.0, le=1.0)
    critical_node_probability: float = Field(..., ge=0.0, le=1.0)
    causal_ate_estimate: float | None = None
    news_trigger: str | None = Field(default=None, description="Headline/snippet that triggered this alert, if any")
    evidence: list[Evidence] = Field(default_factory=list)
    summary: str
    recommended_actions: list[str]
    llm_backend: str

    model_config = {"use_enum_values": True}
