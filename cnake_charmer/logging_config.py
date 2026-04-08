"""Shared logging setup for CLI scripts."""

from __future__ import annotations

import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging with Rich formatting and safe fallback."""
    try:
        from rich.logging import RichHandler
    except ImportError:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
            force=True,
        )
        return

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[
            RichHandler(
                show_time=True,
                show_level=True,
                show_path=False,
                rich_tracebacks=True,
                omit_repeated_times=False,
                markup=False,
            )
        ],
        force=True,
    )
