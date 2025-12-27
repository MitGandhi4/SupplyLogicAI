"""GNN prediction endpoints (critical-node probability + disruption impact)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas import GNNEdgePredictionOut, GNNPredictionOut
from src.graph.build_graph import build_and_save_default_graph, load_graph, GRAPH_PATH
from src.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/gnn", tags=["gnn"])


def _get_graph():
    try:
        return load_graph(GRAPH_PATH)
    except FileNotFoundError:
        return build_and_save_default_graph()


@router.get("/predictions", response_model=list[GNNPredictionOut])
def get_gnn_predictions() -> list[GNNPredictionOut]:
    try:
        from src.gnn.inference import GNNPredictor

        g = _get_graph()
        predictor = GNNPredictor(g)
        results = predictor.predict_all()
    except (RuntimeError, ImportError, ModuleNotFoundError) as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "GNN predictions unavailable: either the model hasn't been trained yet "
                f"(run `python -m src.gnn.train`) or PyTorch/PyTorch Geometric are not installed. ({exc})"
            ),
        ) from exc
    except Exception as exc:
        logger.exception("GNN prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return [GNNPredictionOut(**r) for r in results]


@router.get("/edge-predictions", response_model=list[GNNEdgePredictionOut])
def get_gnn_edge_predictions() -> list[GNNEdgePredictionOut]:
    try:
        from src.gnn.inference import GNNPredictor

        predictions = GNNPredictor(_get_graph()).predict_edges()
    except (RuntimeError, ImportError, ModuleNotFoundError) as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "GNN edge predictions unavailable: either the model hasn't been trained yet "
                f"(run `python -m src.gnn.train`) or PyTorch/PyTorch Geometric are not installed. ({exc})"
            ),
        ) from exc
    except Exception as exc:
        logger.exception("GNN edge prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return [GNNEdgePredictionOut(**prediction) for prediction in predictions]
