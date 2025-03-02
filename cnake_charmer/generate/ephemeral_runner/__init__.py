"""
EphemeralRunner: A package for generating, building, and running code in ephemeral environments.

This package provides tools for:
1. Generating code using LLMs
2. Building and running that code in ephemeral environments
3. Handling errors and regenerating code when needed
"""

import os
import logging
import sys

from cnake_charmer.generate.ephemeral_runner.core import EphemeralCodeGenerator, generate_code
from cnake_charmer.generate.ephemeral_runner.exceptions import (
    EphemeralRunnerError,
    VenvCreationError,
    FileWriteError,
    CompilationError,
    ExecutionError,
    DependencyError,
    ParseError,
)

__version__ = "0.1.0"
__all__ = [
    "EphemeralCodeGenerator",
    "generate_code",
    "EphemeralRunnerError",
    "VenvCreationError",
    "FileWriteError",
    "CompilationError",
    "ExecutionError",
    "DependencyError",
    "ParseError",
]

# Configure log level from environment or default to INFO
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("ephemeral_runner")
logger.setLevel(getattr(logging, log_level))


# Capture uncaught exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        # Let keyboard interrupts pass through
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


# Install the exception handler
sys.excepthook = handle_exception
