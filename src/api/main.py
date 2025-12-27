"""
SupplyMind FastAPI backend.

Run:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routers import alerts, causal, forecast, gnn, graph
from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    logger.info("SupplyMind API starting up (mlflow_tracking_uri=%s)", settings.mlflow_tracking_uri)
    yield
    logger.info("SupplyMind API shutting down")


app = FastAPI(
    title="SupplyMind API",
    description="Supply chain intelligence and resilience platform — graph, GNN, forecasting, causal, and agent endpoints.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(graph.router)
app.include_router(forecast.router)
app.include_router(gnn.router)
app.include_router(causal.router)
app.include_router(alerts.router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health", tags=["health"])
def health() -> dict:
    return {"status": "ok", "service": "supplymind-api"}
