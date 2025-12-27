"""Integration tests for the FastAPI backend using TestClient."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from src.api.main import app  # noqa: E402

client = TestClient(app)


class TestHealth:
    def test_health_check(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestGraphEndpoints:
    def test_get_graph_returns_nodes_and_edges(self):
        resp = client.get("/graph")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["nodes"]) > 0
        assert len(body["edges"]) > 0
        assert {"shipping_cost", "transit_days", "dependency_strength"}.issubset(body["edges"][0])

    def test_get_unknown_node_returns_404(self):
        resp = client.get("/graph/nodes/DOES_NOT_EXIST")
        assert resp.status_code == 404

    def test_critical_nodes_respects_top_k(self):
        resp = client.get("/graph/critical-nodes", params={"top_k": 3})
        assert resp.status_code == 200
        assert len(resp.json()) == 3

    def test_disruption_simulation_unknown_node_returns_404(self):
        resp = client.get("/graph/disruption-simulation/DOES_NOT_EXIST")
        assert resp.status_code == 404


class TestCausalEndpoint:
    def test_causal_effect_returns_ate(self):
        resp = client.get("/causal/effect")
        assert resp.status_code == 200
        assert "ate" in resp.json()


class TestAlertEndpoints:
    def test_alert_history_returns_list(self):
        resp = client.get("/alerts/history")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_generate_alert_returns_structured_alert(self):
        resp = client.post("/alerts/generate", json={"news_query": "supply chain disruption"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["severity"] in {"low", "medium", "high", "critical"}
        assert "recommended_actions" in body
