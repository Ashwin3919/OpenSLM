"""Structured logging setup.

Call ``setup_logging`` once at process startup (in the pipeline's
``configure()`` method).  All subsequent ``logging.getLogger(__name__)``
calls in any module will inherit the configured format and level.
"""

import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", file: Optional[str] = None) -> None:
    """Configure the root logger with a consistent format.

    Safe to call multiple times — uses ``force=True`` so re-configuration
    in tests does not accumulate duplicate handlers.

    Args:
        level: Logging level string (``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
            ``"ERROR"``, ``"CRITICAL"``).
        file: Optional path for a log file.  When provided, log records are
            written to both stdout and the file.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if file:
        handlers.append(logging.FileHandler(file, encoding="utf-8"))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
