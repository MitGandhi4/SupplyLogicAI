"""
Convert the NetworkX supply chain graph into a PyTorch Geometric `Data`
object, including engineered node features and the two training labels used
by the GNN:

  1. `y_critical`  (binary)  — is this node a structural critical failure
                               point (top-quartile composite criticality
                               score from src.graph.graph_utils)?
  2. `y_impact`    (float)   — normalized fraction of total network capacity
                               lost if this node were disrupted, obtained by
                               running the BFS cascade simulation from every
                               node (ground truth for the propagation head).
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from src.graph.graph_utils import compute_centrality_scores, identify_critical_nodes, simulate_disruption_propagation
from src.logging_config import get_logger

logger = get_logger(__name__)

NODE_TYPES = ["supplier", "warehouse", "port", "manufacturer", "retailer"]


def _node_type_one_hot(node_type: str) -> list[float]:
    return [1.0 if node_type == t else 0.0 for t in NODE_TYPES]


def build_node_features(g: nx.DiGraph) -> tuple[list[str], np.ndarray]:
    """Return (ordered node ids, feature matrix) for all nodes in `g`."""
    centrality = compute_centrality_scores(g)
    node_ids = list(g.nodes())
    rows = []
    for n in node_ids:
        d = g.nodes[n]
        c = centrality[n]
        feat = [
            float(d.get("risk_score", 0.0)),
            float(np.log1p(float(d.get("capacity", 0.0)))),
            float(d.get("reliability", 0.0)),
            float(d.get("tier", 0)) / 5.0,
            c["betweenness"],
            c["eigenvector"],
            np.log1p(c["in_degree"]),
            np.log1p(c["out_degree"]),
        ] + _node_type_one_hot(str(d.get("node_type", "")))
        rows.append(feat)
    return node_ids, np.array(rows, dtype=np.float32)


def build_labels(g: nx.DiGraph, node_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Derive (y_critical, y_impact) arrays aligned with `node_ids`."""
    critical_ranked = identify_critical_nodes(g, top_k=max(1, len(node_ids) // 4))
    critical_set = {n for n, _ in critical_ranked}

    total_capacity = sum(float(d.get("capacity", 0.0)) for _, d in g.nodes(data=True)) or 1.0

    y_critical = np.zeros(len(node_ids), dtype=np.float32)
    y_impact = np.zeros(len(node_ids), dtype=np.float32)
    for i, n in enumerate(node_ids):
        y_critical[i] = 1.0 if n in critical_set else 0.0
        sim = simulate_disruption_propagation(g, n)
        y_impact[i] = float(np.clip(sim.total_capacity_lost / total_capacity, 0.0, 1.0))

    return y_critical, y_impact


def graph_to_pyg_data(g: nx.DiGraph, seed: int = 42) -> tuple[Data, list[str]]:
    """Build a PyG `Data` object with node and edge labels and split masks."""
    node_ids, features = build_node_features(g)
    y_critical, y_impact = build_labels(g, node_ids)

    id_to_idx = {n: i for i, n in enumerate(node_ids)}
    edge_index = []
    edge_attr = []
    edge_impact = []
    for u, v, d in g.edges(data=True):
        source_idx, target_idx = id_to_idx[u], id_to_idx[v]
        dependency_strength = float(d.get("dependency_strength", d.get("weight", 0.0)))
        transit_days = float(d.get("transit_days", d.get("lead_time_days", 0.0)))
        shipping_cost = float(d.get("shipping_cost", 0.0))
        volume = float(d.get("volume", 0.0))
        edge_index.append([source_idx, target_idx])
        edge_attr.append([
            dependency_strength,
            float(np.log1p(transit_days) / np.log(31.0)),
            float(np.log1p(shipping_cost) / 15.0),
            float(np.log1p(volume) / 15.0),
        ])
        edge_impact.append(dependency_strength * float(y_impact[target_idx]))

    edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)
    edge_impact_t = torch.tensor(edge_impact, dtype=torch.float32)

    data = Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=edge_index_t,
        edge_attr=edge_attr_t,
        y_critical=torch.tensor(y_critical, dtype=torch.float32),
        y_impact=torch.tensor(y_impact, dtype=torch.float32),
        y_edge_impact=edge_impact_t,
    )

    n = data.num_nodes
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    edge_perm = rng.permutation(data.num_edges)
    n_train_edges = int(0.6 * data.num_edges)
    n_val_edges = int(0.2 * data.num_edges)
    train_edge_mask = torch.zeros(data.num_edges, dtype=torch.bool)
    val_edge_mask = torch.zeros(data.num_edges, dtype=torch.bool)
    test_edge_mask = torch.zeros(data.num_edges, dtype=torch.bool)
    train_edge_mask[edge_perm[:n_train_edges]] = True
    val_edge_mask[edge_perm[n_train_edges:n_train_edges + n_val_edges]] = True
    test_edge_mask[edge_perm[n_train_edges + n_val_edges:]] = True
    data.train_edge_mask = train_edge_mask
    data.val_edge_mask = val_edge_mask
    data.test_edge_mask = test_edge_mask

    logger.info(
        "Built PyG data: %d nodes, %d edges, %d features, %d train/%d val/%d test",
        n, data.num_edges, features.shape[1], int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum()),
    )
    return data, node_ids
