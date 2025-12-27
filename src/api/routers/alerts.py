"""Disruption alert endpoints: trigger the LangGraph agent, browse history."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from src.agent.alert_schema import DisruptionAlert
from src.agent.alert_store import load_alert_history
from src.agent.langgraph_agent import run_agent
from src.api.schemas import AlertRequest
from src.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.post("/generate", response_model=DisruptionAlert)
def generate_alert(request: AlertRequest) -> DisruptionAlert:
    try:
        return run_agent(news_query=request.news_query)
    except Exception as exc:
        logger.exception("Agent failed to generate alert")
        raise HTTPException(status_code=500, detail=f"Agent run failed: {exc}") from exc


@router.get("/history", response_model=list[dict])
def get_alert_history(limit: int = Query(default=20, ge=1, le=200)) -> list[dict]:
    return load_alert_history(limit=limit)
