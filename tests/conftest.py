"""Shared pytest fixtures for the SupplyMind test suite."""
from __future__ import annotations

import networkx as nx
import pytest

from src.graph.build_graph import build_graph
from src.graph.synthetic_data import generate_synthetic_graph_data


@pytest.fixture(scope="session")
def small_graph_data():
    """A small, fast-to-generate synthetic graph for unit tests."""
    return generate_synthetic_graph_data(
        n_suppliers=12, n_warehouses=4, n_ports=3, n_manufacturers=2, n_retailers=2, seed=7,
    )


@pytest.fixture(scope="session")
def small_graph(small_graph_data) -> nx.DiGraph:
    return build_graph(small_graph_data)
