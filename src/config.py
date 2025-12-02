"""
Centralized configuration for SupplyMind.

All secrets and environment-dependent values are loaded from environment
variables (populated from a local .env file via python-dotenv). Nothing in
this module hardcodes secrets.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root if present. Safe to call even if the file
# does not exist (e.g. in CI / Docker where real env vars are injected).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class Settings:
    """Runtime configuration, resolved once at import time."""

    # --- Paths -----------------------------------------------------------
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    vector_store_dir: Path = data_dir / "vector_store"
    model_dir: Path = PROJECT_ROOT / "data" / "models"

    # --- External API keys (never hardcode these) -------------------------
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    newsapi_key: str | None = os.getenv("NEWSAPI_KEY") or os.getenv("NEWS_API_KEY")
    fred_api_key: str | None = os.getenv("FRED_API_KEY")

    # --- LLM / agent configuration ----------------------------------------
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    use_ollama_fallback: bool = os.getenv("USE_OLLAMA_FALLBACK", "false").lower() == "true"

    # --- MLflow -------------------------------------------------------------
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", f"file://{PROJECT_ROOT / 'mlflow' / 'mlruns'}")
    mlflow_experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "supplymind")

    # --- API / dashboard ------------------------------------------------------
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    dashboard_port: int = int(os.getenv("DASHBOARD_PORT", "8501"))

    # --- Graph / synthetic data ----------------------------------------------
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))
    n_suppliers: int = int(os.getenv("N_SUPPLIERS", "80"))
    n_warehouses: int = int(os.getenv("N_WAREHOUSES", "10"))
    n_ports: int = int(os.getenv("N_PORTS", "8"))
    n_manufacturers: int = int(os.getenv("N_MANUFACTURERS", "6"))
    n_retailers: int = int(os.getenv("N_RETAILERS", "12"))

    def ensure_dirs(self) -> None:
        for d in (self.raw_data_dir, self.processed_data_dir, self.vector_store_dir, self.model_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
