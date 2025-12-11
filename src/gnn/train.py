"""
Train the SupplyChainGNN on the supply chain graph and log the run
(hyperparameters, per-epoch metrics, final evaluation, model artifact) to
MLflow.

Run:
    python -m src.gnn.train
"""
from __future__ import annotations

import json

import mlflow
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from src.config import settings
from src.gnn.data_prep import graph_to_pyg_data
from src.gnn.model import SupplyChainGNN
from src.graph.build_graph import build_and_save_default_graph, load_graph, GRAPH_PATH
from src.logging_config import get_logger

logger = get_logger(__name__)

MODEL_PATH = settings.model_dir / "gnn_supply_chain.pt"


def _safe_auc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    y_true_np = y_true.detach().cpu().numpy()
    if len(set(y_true_np.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y_true_np, y_score.detach().cpu().numpy()))


def _f1(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    predictions = (y_score.detach().cpu().numpy() >= 0.5).astype(int)
    return float(f1_score(y_true.detach().cpu().numpy(), predictions, zero_division=0))


def train_gnn(
    epochs: int = 150,
    lr: float = 0.01,
    hidden_channels: int = 64,
    heads: int = 4,
    dropout: float = 0.3,
    weight_decay: float = 5e-4,
) -> dict:
    try:
        g = load_graph(GRAPH_PATH)
    except FileNotFoundError:
        logger.info("No persisted graph found; building synthetic graph now")
        g = build_and_save_default_graph()

    data, node_ids = graph_to_pyg_data(g, seed=settings.random_seed)

    model = SupplyChainGNN(
        in_channels=data.x.shape[1],
        hidden_channels=hidden_channels,
        heads=heads,
        dropout=dropout,
        edge_channels=data.edge_attr.shape[1],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name="gnn_critical_node_propagation"):
        mlflow.log_params({
            "epochs": epochs, "lr": lr, "hidden_channels": hidden_channels,
            "heads": heads, "dropout": dropout, "weight_decay": weight_decay,
            "n_nodes": data.num_nodes, "n_edges": data.num_edges, "n_features": data.x.shape[1],
        })

        best_val_auc = -1.0
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            critical_logits, impact_pred = model(data.x, data.edge_index, data.edge_attr)
            edge_impact_pred = model.predict_edge_impact(data.x, data.edge_index, data.edge_attr)

            loss_critical = F.binary_cross_entropy_with_logits(
                critical_logits[data.train_mask], data.y_critical[data.train_mask]
            )
            loss_impact = F.mse_loss(impact_pred[data.train_mask], data.y_impact[data.train_mask])
            loss_edge_impact = F.mse_loss(
                edge_impact_pred[data.train_edge_mask], data.y_edge_impact[data.train_edge_mask],
            )
            loss = loss_critical + loss_impact + loss_edge_impact
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_logits, val_impact = model(data.x, data.edge_index, data.edge_attr)
                val_edge_impact = model.predict_edge_impact(data.x, data.edge_index, data.edge_attr)
                val_critical_prob = torch.sigmoid(val_logits[data.val_mask])
                val_auc = _safe_auc(data.y_critical[data.val_mask], val_critical_prob)
                val_f1 = _f1(data.y_critical[data.val_mask], val_critical_prob)
                val_mse = F.mse_loss(val_impact[data.val_mask], data.y_impact[data.val_mask]).item()
                val_edge_mse = F.mse_loss(
                    val_edge_impact[data.val_edge_mask], data.y_edge_impact[data.val_edge_mask],
                ).item()

            if epoch % 10 == 0 or epoch == epochs:
                logger.info(
                    "epoch %03d | train_loss=%.4f (crit=%.4f node_impact=%.4f edge_impact=%.4f) | val_f1=%.4f val_auc=%.4f val_node_mse=%.4f val_edge_mse=%.4f",
                    epoch, loss.item(), loss_critical.item(), loss_impact.item(), loss_edge_impact.item(), val_f1, val_auc, val_mse, val_edge_mse,
                )

            mlflow.log_metrics({
                "train_loss": loss.item(),
                "train_loss_critical": loss_critical.item(),
                "train_loss_impact": loss_impact.item(),
                "train_loss_edge_impact": loss_edge_impact.item(),
                "val_auc": val_auc if val_auc == val_auc else 0.0,  # nan guard
                "val_f1": val_f1,
                "val_mse": val_mse,
                "val_edge_mse": val_edge_mse,
            }, step=epoch)

            if val_auc == val_auc and val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), MODEL_PATH)

        # Final test-set evaluation.
        model.eval()
        with torch.no_grad():
            test_logits, test_impact = model(data.x, data.edge_index, data.edge_attr)
            test_edge_impact = model.predict_edge_impact(data.x, data.edge_index, data.edge_attr)
            test_critical_prob = torch.sigmoid(test_logits[data.test_mask])
            test_auc = _safe_auc(data.y_critical[data.test_mask], test_critical_prob)
            test_f1 = _f1(data.y_critical[data.test_mask], test_critical_prob)
            test_mse = F.mse_loss(test_impact[data.test_mask], data.y_impact[data.test_mask]).item()
            test_edge_mse = F.mse_loss(
                test_edge_impact[data.test_edge_mask], data.y_edge_impact[data.test_edge_mask],
            ).item()

        metrics = {
            "test_auc": test_auc if test_auc == test_auc else 0.0,
            "test_f1": test_f1,
            "test_mse": test_mse,
            "test_edge_mse": test_edge_mse,
            "best_val_auc": max(best_val_auc, 0.0),
        }
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(MODEL_PATH)) if MODEL_PATH.exists() else None

        meta = {
            "in_channels": data.x.shape[1],
            "hidden_channels": hidden_channels,
            "heads": heads,
            "edge_channels": data.edge_attr.shape[1],
            "node_ids": node_ids,
        }
        meta_path = settings.model_dir / "gnn_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        mlflow.log_artifact(str(meta_path))

        logger.info("GNN training complete. test_auc=%.4f test_mse=%.4f", metrics["test_auc"], metrics["test_mse"])
        return metrics


if __name__ == "__main__":
    train_gnn()
