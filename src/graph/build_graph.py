"""
Build and persist the supply chain NetworkX graph from validated Pydantic
node/edge data (either the synthetic generator or an ingested Kaggle CSV
mapped into the same schema).
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from src.config import settings
from src.graph.schema import SupplyChainGraphData
from src.graph.synthetic_data import generate_synthetic_graph_data
from src.logging_config import get_logger

logger = get_logger(__name__)

GRAPH_PATH = settings.processed_data_dir / "supply_chain_graph.graphml"
GRAPH_JSON_PATH = settings.processed_data_dir / "supply_chain_graph.json"


def build_graph(data: SupplyChainGraphData) -> nx.DiGraph:
    """Construct a directed NetworkX graph from validated nodes/edges."""
    g = nx.DiGraph()

    for node in data.nodes:
        g.add_node(node.node_id, **node.model_dump(exclude={"node_id"}))

    for edge in data.edges:
        g.add_edge(edge.source, edge.target, **edge.model_dump(exclude={"source", "target"}))

    logger.info("Built graph with %d nodes and %d edges", g.number_of_nodes(), g.number_of_edges())
    return g


def save_graph(g: nx.DiGraph, path: Path = GRAPH_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # GraphML requires scalar attribute types; our schema is already scalar-only.
    nx.write_graphml(g, path)
    # Also dump a plain JSON (node-link format) for easy consumption by the
    # FastAPI backend / Streamlit dashboard without a NetworkX dependency.
    data = nx.node_link_data(g)
    with open(GRAPH_JSON_PATH, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved graph to %s and %s", path, GRAPH_JSON_PATH)


def load_graph(path: Path = GRAPH_PATH) -> nx.DiGraph:
    if not path.exists():
        raise FileNotFoundError(
            f"Graph file not found at {path}. Run `python -m src.graph.build_graph` first."
        )
    g = nx.read_graphml(path)
    logger.info("Loaded graph from %s: %d nodes, %d edges", path, g.number_of_nodes(), g.number_of_edges())
    return g


def build_and_save_default_graph() -> nx.DiGraph:
    """Convenience entrypoint: generate synthetic data, build, and persist the graph."""
    data = generate_synthetic_graph_data()
    g = build_graph(data)
    save_graph(g)
    return g


if __name__ == "__main__":
    build_and_save_default_graph()
