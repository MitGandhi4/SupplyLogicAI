"""
Shared logging configuration. Every module in SupplyMind uses
`logging.getLogger(__name__)` — no `print()` statements anywhere.

Usage:
    from src.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("message")
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False
_LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "logs"


def configure_logging(level: int = logging.INFO) -> None:
    """Idempotently configure the root logger with console + file handlers."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    file_handler = logging.FileHandler(_LOG_DIR / "supplymind.log")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # Quiet down noisy third-party loggers.
    for noisy in ("urllib3", "httpx", "matplotlib", "prophet", "cmdstanpy"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
