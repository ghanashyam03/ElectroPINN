"""Structured logging configuration."""

from __future__ import annotations

import logging
import sys
from typing import Any


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a consistent format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)


def log_config(logger: logging.Logger, config: dict[str, Any]) -> None:
    """Log a configuration dictionary."""
    for key, value in config.items():
        logger.info("%s: %s", key, value)
