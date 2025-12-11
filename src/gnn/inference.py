"""
Load a trained SupplyChainGNN and expose simple prediction helpers used by
the FastAPI backend and the LangGraph agent.
"""
from __future__ import annotations

import json

import networkx as nx
import torch

from src.config import settings
from src.gnn.data_prep import graph_to_pyg_data
from src.gnn.model import SupplyChainGNN
from src.gnn.train import MODEL_PATH
from src.logging_config import get_logger

logger = get_logger(__name__)

META_PATH = settings.model_dir / "gnn_meta.json"


class GNNPredictor:
    """Wraps a trained GNN + the graph it was trained on for inference."""

    def __init__(self, g: nx.DiGraph):
        self.g = g
        self.data, self.node_ids = graph_to_pyg_data(g, seed=settings.random_seed)
        self.model: SupplyChainGNN | None = None
        self._load_model()

    def _load_model(self) -> None:
        if not MODEL_PATH.exists() or not META_PATH.exists():
            logger.warning("No trained GNN found at %s; predictions will be unavailable until training runs", MODEL_PATH)
            return
        with open(META_PATH) as f:
            meta = json.load(f)
        if meta.get("edge_channels") != self.data.edge_attr.shape[1]:
            logger.warning(
                "Saved GNN metadata predates edge-impact modeling or uses incompatible edge features; retrain the model"
            )
            return
        model = SupplyChainGNN(
            in_channels=meta["in_channels"],
            hidden_channels=meta["hidden_channels"],
            heads=meta["heads"],
            edge_channels=meta.get("edge_channels", self.data.edge_attr.shape[1]),
        )
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        except RuntimeError as exc:
            logger.warning("Saved GNN checkpoint is incompatible with this model version: %s", exc)
            return
        model.eval()
        self.model = model
        logger.info("Loaded trained GNN from %s", MODEL_PATH)

    def predict_all(self) -> list[dict]:
        """Return per-node critical probability + predicted disruption impact."""
        if self.model is None:
            raise RuntimeError("GNN model is not trained/loaded. Run `python -m src.gnn.train` first.")
        crit_prob, impact = self.model.predict(self.data.x, self.data.edge_index, self.data.edge_attr)
        results = []
        for i, node_id in enumerate(self.node_ids):
            results.append({
                "node_id": node_id,
                "critical_probability": float(crit_prob[i]),
                "predicted_impact": float(impact[i]),
            })
        results.sort(key=lambda r: r["critical_probability"], reverse=True)
        return results

    def predict_edges(self) -> list[dict]:
        """Return predicted propagation impact for every directed edge."""
        if self.model is None:
            raise RuntimeError("GNN model is not trained/loaded. Run `python -m src.gnn.train` first.")
        with torch.no_grad():
            scores = self.model.predict_edge_impact(self.data.x, self.data.edge_index, self.data.edge_attr)
        return [
            {
                "source": self.node_ids[int(self.data.edge_index[0, index])],
                "target": self.node_ids[int(self.data.edge_index[1, index])],
                "predicted_impact": float(score),
            }
            for index, score in enumerate(scores)
        ]

    def predict_node(self, node_id: str) -> dict:
        results = self.predict_all()
        for r in results:
            if r["node_id"] == node_id:
                return r
        raise KeyError(f"Node '{node_id}' not found in GNN predictions")
