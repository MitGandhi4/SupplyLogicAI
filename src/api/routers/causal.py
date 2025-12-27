"""Causal effect endpoint (cached DoWhy ATE estimate)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas import CausalEffectOut
from src.causal.causal_model import get_or_compute_causal_ate
from src.graph.build_graph import build_and_save_default_graph, load_graph, GRAPH_PATH
from src.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/causal", tags=["causal"])


@router.get("/effect", response_model=CausalEffectOut)
def get_causal_effect() -> CausalEffectOut:
    try:
        g = load_graph(GRAPH_PATH)
    except FileNotFoundError:
        g = build_and_save_default_graph()

    try:
        result = get_or_compute_causal_ate(g)
    except Exception as exc:
        logger.exception("Causal effect estimation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return CausalEffectOut(
        ate=result.ate,
        confidence_interval=list(result.confidence_interval) if result.confidence_interval else None,
        placebo_refutation_passed=result.placebo_refutation_passed,
        random_common_cause_refutation_passed=result.random_common_cause_refutation_passed,
        summary=result.summary,
    )
