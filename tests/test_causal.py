"""Tests for the DoWhy causal ML pipeline."""
from __future__ import annotations

import pytest

pytest.importorskip("dowhy")

from src.causal.causal_model import build_causal_dataset, estimate_disruption_causal_effect  # noqa: E402


class TestCausalDataset:
    def test_dataset_has_expected_columns(self, small_graph):
        df = build_causal_dataset(small_graph, n_trials_per_node=3)
        expected_cols = {"disrupted", "downstream_impact", "risk_score", "reliability", "capacity_log", "tier", "betweenness"}
        assert expected_cols.issubset(df.columns)

    def test_dataset_row_count(self, small_graph):
        df = build_causal_dataset(small_graph, n_trials_per_node=4)
        assert len(df) == small_graph.number_of_nodes() * 4

    def test_treatment_is_binary(self, small_graph):
        df = build_causal_dataset(small_graph, n_trials_per_node=3)
        assert set(df["disrupted"].unique()).issubset({0, 1})


class TestCausalEstimation:
    def test_estimate_runs_and_returns_result(self, small_graph):
        df = build_causal_dataset(small_graph, n_trials_per_node=6)
        result = estimate_disruption_causal_effect(df)
        assert isinstance(result.ate, float)
        assert result.summary
        # A disruption should not have a strongly negative effect on impact.
        assert result.ate > -0.5
