"""
Abstract base builder for code execution.

This module defines the base class for all builders that handle
code execution in ephemeral environments.
"""

import abc
import logging
import os
import re
import sys
import tempfile
import subprocess
from typing import List, Optional, Set, Tuple, Dict, Any

# Configure logger
logger = logging.getLogger("ephemeral_runner.builders")


class BaseBuilder(abc.ABC):
    """
    Abstract base class for all code builders.

    Builders are responsible for:
    1. Creating ephemeral environments
    2. Detecting dependencies
    3. Installing dependencies
    4. Building and executing code
    """

    def __init__(self, request_id: str = None):
        """
        Initialize the builder.

        Args:
            request_id: Unique identifier for this build request
        """
        self.request_id = request_id or id(self)

    @abc.abstractmethod
    def build_and_run(self, code_str: str) -> Optional[str]:
        """
        Build and run the given code.

        Args:
            code_str: The code to build and run

        Returns:
            Error message if failed, None if successful
        """
        pass

    def run_in_venv(
        self, venv_dir: str, command: str, cwd: str = None, capture_stdout: bool = False
    ) -> Optional[str]:
        """
        Run a shell command inside the virtual environment.

        Args:
            venv_dir: Path to the virtual environment
            command: Command to run
            cwd: Working directory for the command
            capture_stdout: Whether to capture and return stdout

        Returns:
            None if successful, error string if failed,
            or stdout if successful and capture_stdout is True
        """
        # Determine the Python executable path in the virtual environment
        if sys.platform == "win32":
            bin_dir = os.path.join(venv_dir, "Scripts")
        else:
            bin_dir = os.path.join(venv_dir, "bin")

        # Parse command to determine what executable to use
        if command.startswith("pip "):
            exe = os.path.join(bin_dir, "pip")
            args = command.split(" ", 1)[1].split()
        elif command.startswith("python "):
            exe = os.path.join(bin_dir, "python")
            args = command.split(" ", 1)[1].split()
        else:
            # Default to python for other commands
            exe = os.path.join(bin_dir, "python")
            args = command.split()

        real_cmd = [exe] + args
        cmd_str = " ".join(real_cmd)

        # Check if the executable exists
        if not os.path.exists(exe):
            logger.error(f"Request {self.request_id}: Executable not found: {exe}")
            return f"Command execution failed: executable not found at {exe}"

        logger.debug(
            f"Request {self.request_id}: Running in venv: {cmd_str} (cwd={cwd}, capture_stdout={capture_stdout})"
        )

        try:
            proc = subprocess.run(
                real_cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
            )

            # Always log some basic info about the execution
            logger.debug(
                f"Request {self.request_id}: Command returned with code {proc.returncode}"
            )

            if proc.returncode != 0:
                error_output = f"Command failed with code {proc.returncode}\n"
                if proc.stdout:
                    error_output += f"STDOUT:\n{proc.stdout}\n"
                if proc.stderr:
                    error_output += f"STDERR:\n{proc.stderr}"
                logger.error(
                    f"Request {self.request_id}: Command failed: {cmd_str} (code={proc.returncode})"
                )
                logger.debug(
                    f"Request {self.request_id}: Error output lengths - stdout: {len(proc.stdout) if proc.stdout else 0}, stderr: {len(proc.stderr) if proc.stderr else 0}"
                )
                return error_output.strip()

            # Return stdout if requested
            if capture_stdout and proc.stdout:
                logger.debug(
                    f"Request {self.request_id}: Command captured stdout ({len(proc.stdout)} chars)"
                )
                return proc.stdout.strip()

            # Otherwise just log the stdout for debugging
            elif proc.stdout and proc.stdout.strip():
                stdout_snippet = (
                    proc.stdout[:300] + "..."
                    if len(proc.stdout) > 300
                    else proc.stdout.strip()
                )
                logger.debug(
                    f"Request {self.request_id}: Command output length: {len(proc.stdout)}, snippet: {stdout_snippet}"
                )

        except Exception as e:
            logger.error(
                f"Request {self.request_id}: Exception executing command: {str(e)}"
            )
            return f"Command execution failed with exception: {str(e)}"

        return None

    def parse_dependencies(self, code_str: str) -> List[str]:
        """
        Parse dependencies from code.

        Args:
            code_str: Code to parse

        Returns:
            List of dependency names
        """
        from ephemeral_runner.utils.dependencies import parse_imports, detect_cython

        # Use the utility function to parse dependencies
        is_cython = self.is_cython(code_str)
        return parse_imports(code_str, is_cython)

    def is_cython(self, code_str: str) -> bool:
        """
        Check if code is Cython based on key indicators.

        Args:
            code_str: Code to check

        Returns:
            True if code is Cython, False otherwise
        """
        from ephemeral_runner.utils.dependencies import detect_cython

        # Use the utility function to detect Cython
        is_cython = detect_cython(code_str)
        if is_cython:
            logger.debug(f"Identified code as Cython")
        return is_cython
