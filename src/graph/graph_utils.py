"""
Graph analytics: centrality-based critical node detection and cascading
disruption-propagation simulation. These utilities are used both (a) to
generate training labels for the GNN, and (b) directly by the FastAPI
backend / agent for explainable, non-ML fallback answers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from src.logging_config import get_logger

logger = get_logger(__name__)


def compute_centrality_scores(g: nx.DiGraph) -> dict[str, dict[str, float]]:
    """Compute betweenness, degree, and eigenvector centrality for every node."""
    betweenness = nx.betweenness_centrality(g, weight="weight")
    in_degree = dict(g.in_degree())
    out_degree = dict(g.out_degree())
    try:
        eigenvector = nx.eigenvector_centrality_numpy(g)
    except Exception as exc:  # pragma: no cover - numerical edge case
        logger.warning("Eigenvector centrality failed (%s); falling back to zeros", exc)
        eigenvector = {n: 0.0 for n in g.nodes()}

    scores: dict[str, dict[str, float]] = {}
    for n in g.nodes():
        scores[n] = {
            "betweenness": betweenness.get(n, 0.0),
            "in_degree": float(in_degree.get(n, 0)),
            "out_degree": float(out_degree.get(n, 0)),
            "eigenvector": eigenvector.get(n, 0.0),
        }
    return scores


def identify_critical_nodes(g: nx.DiGraph, top_k: int = 10) -> list[tuple[str, float]]:
    """Rank nodes by a composite criticality score (betweenness + risk + inverse reliability)."""
    centrality = compute_centrality_scores(g)
    max_betweenness = max((v["betweenness"] for v in centrality.values()), default=1.0) or 1.0

    ranked = []
    for n, data in g.nodes(data=True):
        norm_betweenness = centrality[n]["betweenness"] / max_betweenness
        risk = float(data.get("risk_score", 0.0))
        reliability = float(data.get("reliability", 1.0))
        composite = 0.5 * norm_betweenness + 0.3 * risk + 0.2 * (1 - reliability)
        ranked.append((n, composite))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


@dataclass
class DisruptionSimulationResult:
    origin_node: str
    affected_nodes: list[str] = field(default_factory=list)
    propagation_steps: list[list[str]] = field(default_factory=list)
    total_capacity_lost: float = 0.0


def simulate_disruption_propagation(
    g: nx.DiGraph,
    origin_node: str,
    propagation_prob_threshold: float = 0.3,
    max_hops: int = 6,
) -> DisruptionSimulationResult:
    """Simulate a cascading disruption starting at `origin_node`.

    A downstream neighbor is considered "affected" if the edge weight
    (dependency strength) exceeds `propagation_prob_threshold`, modeling the
    intuition that only sufficiently strong dependencies transmit failures.
    This BFS-based cascade is intentionally simple/interpretable and serves
    as ground truth for training the GNN's propagation-prediction head.
    """
    if origin_node not in g:
        raise ValueError(f"Node '{origin_node}' not found in graph")

    visited = {origin_node}
    frontier = [origin_node]
    steps: list[list[str]] = []
    total_capacity_lost = float(g.nodes[origin_node].get("capacity", 0.0))

    for _hop in range(max_hops):
        next_frontier = []
        for node in frontier:
            for neighbor in g.successors(node):
                if neighbor in visited:
                    continue
                edge_data = g.get_edge_data(node, neighbor) or {}
                weight = float(edge_data.get("weight", 0.0))
                if weight >= propagation_prob_threshold:
                    visited.add(neighbor)
                    next_frontier.append(neighbor)
                    total_capacity_lost += float(g.nodes[neighbor].get("capacity", 0.0))
        if not next_frontier:
            break
        steps.append(next_frontier)
        frontier = next_frontier

    result = DisruptionSimulationResult(
        origin_node=origin_node,
        affected_nodes=sorted(visited - {origin_node}),
        propagation_steps=steps,
        total_capacity_lost=total_capacity_lost,
    )
    logger.info(
        "Disruption at %s propagated to %d nodes over %d hops (capacity lost: %.1f)",
        origin_node, len(result.affected_nodes), len(steps), total_capacity_lost,
    )
    return result


def single_points_of_failure(g: nx.DiGraph) -> list[str]:
    """Nodes whose removal disconnects at least one manufacturer from all its suppliers."""
    manufacturers = [n for n, d in g.nodes(data=True) if d.get("node_type") == "manufacturer"]
    spofs = []
    undirected = g.to_undirected()
    for node in list(g.nodes()):
        test_graph = undirected.copy()
        test_graph.remove_node(node)
        for mfg in manufacturers:
            if mfg == node or mfg not in test_graph:
                continue
            reachable = nx.node_connected_component(test_graph, mfg) if mfg in test_graph else set()
            upstream_before = nx.ancestors(g, mfg) if mfg in g else set()
            # If a manufacturer loses all upstream connectivity after removing `node`.
            if upstream_before and not (reachable - {mfg}):
                spofs.append(node)
                break
    return spofs
