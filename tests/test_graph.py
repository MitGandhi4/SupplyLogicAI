"""Tests for graph schema validation, construction, and analytics."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.graph.graph_utils import identify_critical_nodes, simulate_disruption_propagation
from src.graph.schema import Edge, EdgeType, Node, NodeType, SupplyChainGraphData


def _valid_node(node_id="A", **overrides) -> Node:
    defaults = dict(
        node_id=node_id, node_type=NodeType.SUPPLIER, name="Test Supplier", country="USA",
        region="AMER", latitude=10.0, longitude=10.0, risk_score=0.3, capacity=100.0,
        reliability=0.9, tier=1,
    )
    defaults.update(overrides)
    return Node(**defaults)


class TestNodeSchema:
    def test_valid_node_passes(self):
        node = _valid_node()
        assert node.node_id == "A"

    def test_risk_score_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            _valid_node(risk_score=1.5)

    def test_negative_capacity_rejected(self):
        with pytest.raises(ValidationError):
            _valid_node(capacity=-5.0)

    def test_invalid_node_type_rejected(self):
        with pytest.raises(ValidationError):
            Node(node_id="A", node_type="not_a_type", name="x", country="US", latitude=0, longitude=0,
                 risk_score=0.1, capacity=1.0, reliability=0.5)


class TestEdgeSchema:
    def test_self_loop_rejected(self):
        with pytest.raises(ValidationError):
            Edge(source="A", target="A", edge_type=EdgeType.SHIPS_TO, weight=0.5, lead_time_days=1.0, volume=1.0)

    def test_weight_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            Edge(source="A", target="B", edge_type=EdgeType.SHIPS_TO, weight=1.2, lead_time_days=1.0, volume=1.0)


class TestSupplyChainGraphData:
    def test_duplicate_node_ids_rejected(self):
        with pytest.raises(ValidationError):
            SupplyChainGraphData(nodes=[_valid_node("A"), _valid_node("A")], edges=[])

    def test_dangling_edge_reference_rejected(self):
        with pytest.raises(ValidationError):
            SupplyChainGraphData(
                nodes=[_valid_node("A")],
                edges=[Edge(source="A", target="GHOST", edge_type=EdgeType.SHIPS_TO, weight=0.5, lead_time_days=1.0, volume=1.0)],
            )

    def test_valid_graph_data_passes(self):
        data = SupplyChainGraphData(
            nodes=[_valid_node("A"), _valid_node("B")],
            edges=[Edge(source="A", target="B", edge_type=EdgeType.SHIPS_TO, weight=0.5, lead_time_days=1.0, volume=1.0)],
        )
        assert len(data.nodes) == 2
        assert len(data.edges) == 1

    def test_edge_operational_features_default_from_legacy_fields(self):
        edge = Edge(
            source="A", target="B", edge_type=EdgeType.SHIPS_TO, weight=0.5,
            lead_time_days=4.0, volume=10.0,
        )
        assert edge.transit_days == 4.0
        assert edge.dependency_strength == 0.5
        assert edge.shipping_cost == 0.0


class TestGraphAnalytics:
    def test_graph_has_expected_node_count(self, small_graph):
        assert small_graph.number_of_nodes() == 12 + 4 + 3 + 2 + 2

    def test_identify_critical_nodes_returns_requested_count(self, small_graph):
        ranked = identify_critical_nodes(small_graph, top_k=5)
        assert len(ranked) == 5
        # Scores should be sorted descending.
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_disruption_simulation_runs_for_every_node(self, small_graph):
        for node in list(small_graph.nodes())[:5]:
            result = simulate_disruption_propagation(small_graph, node)
            assert result.origin_node == node
            assert result.total_capacity_lost >= 0

    def test_disruption_simulation_unknown_node_raises(self, small_graph):
        with pytest.raises(ValueError):
            simulate_disruption_propagation(small_graph, "DOES_NOT_EXIST")


def test_default_graph_has_all_node_types_and_no_orphans():
    import networkx as nx

    from src.graph.build_graph import build_graph
    from src.graph.synthetic_data import generate_synthetic_graph_data

    graph = build_graph(generate_synthetic_graph_data(
        n_suppliers=80,
        n_warehouses=10,
        n_ports=8,
        n_manufacturers=6,
        n_retailers=12,
        seed=42,
    ))
    assert graph.number_of_nodes() >= 100
    assert {data["node_type"] for _, data in graph.nodes(data=True)} == {
        "supplier", "manufacturer", "warehouse", "port", "retailer",
    }
    assert list(nx.isolates(graph)) == []
