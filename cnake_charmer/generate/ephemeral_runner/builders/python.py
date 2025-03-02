"""
Python code builder for ephemeral environments.

This module provides a builder for executing Python code in ephemeral environments.
"""

import os
import tempfile
import venv
import logging
from typing import List, Optional

from ephemeral_runner.builders.base import BaseBuilder
from ephemeral_runner.utils.venv import create_venv
from ephemeral_runner.exceptions import VenvCreationError, FileWriteError

# Configure logger
logger = logging.getLogger("ephemeral_runner.builders.python")


class PythonBuilder(BaseBuilder):
    """
    Builder for Python code execution in ephemeral environments.

    This builder:
    1. Creates a virtual environment
    2. Installs dependencies
    3. Runs Python code
    """

    def __init__(self, request_id: str = None):
        """
        Initialize the Python builder.

        Args:
            request_id: Unique identifier for this build request
        """
        super().__init__(request_id)

    def build_and_run(self, code_str: str) -> Optional[str]:
        """
        Build and run Python code in an ephemeral environment.

        Args:
            code_str: Python code to run

        Returns:
            Error message if failed, None if successful
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1) create ephemeral venv
            logger.info(
                f"Request {self.request_id}: Creating ephemeral venv for Python execution in {tmpdir}"
            )
            venv_dir = os.path.join(tmpdir, "venv")
            try:
                venv.create(venv_dir, with_pip=True)
                logger.debug(
                    f"Request {self.request_id}: Successfully created venv at {venv_dir}"
                )
            except Exception as e:
                logger.error(
                    f"Request {self.request_id}: Error creating venv: {str(e)}"
                )
                return f"Failed to create virtual environment: {str(e)}"

            # 2) parse dependencies (imported libs) -> pip install
            try:
                deps = self.parse_dependencies(code_str)
                logger.info(
                    f"Request {self.request_id}: Detected dependencies: {deps} (count: {len(deps)})"
                )
            except Exception as e:
                logger.error(
                    f"Request {self.request_id}: Error parsing dependencies: {str(e)}"
                )
                return f"Failed to parse dependencies: {str(e)}"

            commands = [
                # upgrade pip
                f"pip install --upgrade pip wheel setuptools",
            ]
            if deps:
                commands.append(f"pip install {' '.join(deps)}")

            # 3) write code to .py
            py_path = os.path.join(tmpdir, "gen_code.py")
            try:
                with open(py_path, "w") as f:
                    f.write(code_str)
                logger.info(
                    f"Request {self.request_id}: Wrote Python code ({len(code_str)} bytes) to {py_path}"
                )
            except Exception as e:
                logger.error(
                    f"Request {self.request_id}: Error writing code to file: {str(e)}"
                )
                return f"Failed to write code to file: {str(e)}"

            # 4) run
            for i, cmd in enumerate(commands):
                logger.info(
                    f"Request {self.request_id}: Running command [{i+1}/{len(commands)}]: {cmd}"
                )
                err = self.run_in_venv(venv_dir, cmd)
                if err:
                    logger.error(
                        f"Request {self.request_id}: Dependency installation failed: {err[:100]}..."
                        if len(err) > 100
                        else err
                    )
                    return f"Python ephemeral venv install error: {err}"

            logger.info(f"Request {self.request_id}: Executing Python code")
            run_cmd = f"python {py_path}"
            err = self.run_in_venv(venv_dir, run_cmd)
            if err:
                logger.error(
                    f"Request {self.request_id}: Python execution failed: {err[:100]}..."
                    if len(err) > 100
                    else err
                )
                return f"Python run error: {err}"

            logger.info(
                f"Request {self.request_id}: Python execution completed successfully"
            )
        return None
