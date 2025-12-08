"""
Tests for the GNN data prep and model. Skipped automatically if PyTorch /
PyTorch Geometric are not installed (e.g. disk-constrained CI sandboxes) —
these run fully on a normal development machine with `pip install -r
requirements.txt`.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from src.gnn.data_prep import graph_to_pyg_data  # noqa: E402
from src.gnn.model import SupplyChainGNN  # noqa: E402


class TestDataPrep:
    def test_pyg_data_shapes(self, small_graph):
        data, node_ids = graph_to_pyg_data(small_graph)
        assert data.num_nodes == small_graph.number_of_nodes()
        assert data.num_nodes == len(node_ids)
        assert data.x.shape[0] == data.num_nodes
        assert data.y_critical.shape[0] == data.num_nodes
        assert data.y_impact.shape[0] == data.num_nodes
        assert data.y_edge_impact.shape[0] == data.num_edges
        assert data.edge_attr.shape == (data.num_edges, 4)
        assert (data.train_edge_mask.sum() + data.val_edge_mask.sum() + data.test_edge_mask.sum()).item() == data.num_edges

    def test_masks_partition_nodes(self, small_graph):
        data, _ = graph_to_pyg_data(small_graph)
        total = (data.train_mask.sum() + data.val_mask.sum() + data.test_mask.sum()).item()
        assert total == data.num_nodes
        overlap = (data.train_mask & data.val_mask).sum().item()
        assert overlap == 0


class TestModel:
    def test_forward_pass_shapes(self, small_graph):
        data, _ = graph_to_pyg_data(small_graph)
        model = SupplyChainGNN(in_channels=data.x.shape[1], hidden_channels=8, heads=2)
        critical_logits, impact = model(data.x, data.edge_index)
        assert critical_logits.shape == (data.num_nodes,)
        assert impact.shape == (data.num_nodes,)
        assert torch.all((impact >= 0) & (impact <= 1))

    def test_predict_returns_probabilities(self, small_graph):
        data, _ = graph_to_pyg_data(small_graph)
        model = SupplyChainGNN(in_channels=data.x.shape[1], hidden_channels=8, heads=2)
        probs, impact = model.predict(data.x, data.edge_index, data.edge_attr)
        assert torch.all((probs >= 0) & (probs <= 1))
        assert torch.all((impact >= 0) & (impact <= 1))

    def test_edge_impact_predictions_are_bounded(self, small_graph):
        data, _ = graph_to_pyg_data(small_graph)
        model = SupplyChainGNN(in_channels=data.x.shape[1], hidden_channels=8, heads=2)
        edge_impact = model.predict_edge_impact(data.x, data.edge_index, data.edge_attr)
        assert edge_impact.shape == (data.num_edges,)
        assert torch.all((edge_impact >= 0) & (edge_impact <= 1))
