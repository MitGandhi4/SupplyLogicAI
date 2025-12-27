"""Graph query endpoints: full graph, node lookup, critical nodes, disruption simulation."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import CriticalNodeOut, DisruptionSimulationOut, EdgeOut, GraphOut, NodeOut
from src.graph.build_graph import build_and_save_default_graph, load_graph, GRAPH_PATH
from src.graph.graph_utils import identify_critical_nodes, simulate_disruption_propagation
from src.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/graph", tags=["graph"])


def _get_graph():
    try:
        return load_graph(GRAPH_PATH)
    except FileNotFoundError:
        logger.info("No persisted graph found; building synthetic default graph")
        return build_and_save_default_graph()


@router.get("", response_model=GraphOut)
def get_graph() -> GraphOut:
    try:
        g = _get_graph()
        nodes = [NodeOut(node_id=n, **{k: v for k, v in d.items()}) for n, d in g.nodes(data=True)]
        edges = [EdgeOut(source=u, target=v, **{k: val for k, val in d.items()}) for u, v, d in g.edges(data=True)]
        return GraphOut(nodes=nodes, edges=edges)
    except Exception as exc:
        logger.exception("Failed to build graph response")
        raise HTTPException(status_code=500, detail=f"Failed to load graph: {exc}") from exc


@router.get("/nodes/{node_id}", response_model=NodeOut)
def get_node(node_id: str) -> NodeOut:
    g = _get_graph()
    if node_id not in g:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
    return NodeOut(node_id=node_id, **g.nodes[node_id])


@router.get("/critical-nodes", response_model=list[CriticalNodeOut])
def get_critical_nodes(top_k: int = Query(default=10, ge=1, le=200)) -> list[CriticalNodeOut]:
    g = _get_graph()
    ranked = identify_critical_nodes(g, top_k=top_k)
    return [CriticalNodeOut(node_id=n, composite_score=s) for n, s in ranked]


@router.get("/disruption-simulation/{node_id}", response_model=DisruptionSimulationOut)
def get_disruption_simulation(node_id: str) -> DisruptionSimulationOut:
    g = _get_graph()
    if node_id not in g:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
    try:
        result = simulate_disruption_propagation(g, node_id)
    except Exception as exc:
        logger.exception("Disruption simulation failed for node %s", node_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return DisruptionSimulationOut(
        origin_node=result.origin_node,
        affected_nodes=result.affected_nodes,
        n_hops=len(result.propagation_steps),
        total_capacity_lost=result.total_capacity_lost,
    )
