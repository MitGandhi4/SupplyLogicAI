"""Append-only JSONL persistence for agent-generated alerts (used by the
dashboard's 'agent reasoning log' and the FastAPI /alerts/history endpoint)."""
from __future__ import annotations

import json

from src.agent.alert_schema import DisruptionAlert
from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)

ALERT_HISTORY_PATH = settings.processed_data_dir / "alerts_history.jsonl"


def append_alert(alert: DisruptionAlert) -> None:
    ALERT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(ALERT_HISTORY_PATH, "a") as f:
            f.write(alert.model_dump_json() + "\n")
    except OSError:
        logger.exception("Failed to persist alert %s to history", alert.alert_id)


def load_alert_history(limit: int = 50) -> list[dict]:
    if not ALERT_HISTORY_PATH.exists():
        return []
    alerts = []
    try:
        with open(ALERT_HISTORY_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    alerts.append(json.loads(line))
    except OSError:
        logger.exception("Failed to read alert history")
        return []
    return list(reversed(alerts))[:limit]
