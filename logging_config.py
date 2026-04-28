"""Logging configuration for the project."""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with a readable console handler.

    Call this once at the start of any script or entry point.

    Args:
        level: Logging level (default: INFO).
    """
    root = logging.getLogger()

    # Avoid adding duplicate handlers if called multiple times
    if root.handlers:
        return

    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
