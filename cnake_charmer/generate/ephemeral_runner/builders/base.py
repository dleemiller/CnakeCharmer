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
import traceback
from typing import List, Optional, Set, Tuple, Dict, Any, Union

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

    def __init__(self, request_id: str = None, verbose_logging: bool = True):
        """
        Initialize the builder.

        Args:
            request_id: Unique identifier for this build request
            verbose_logging: Whether to enable detailed logging
        """
        self.request_id = request_id or id(self)
        self.verbose_logging = verbose_logging
        self.log_prefix = f"Request {self.request_id}: "

    def log(self, level: str, message: str, exc_info: bool = False):
        """
        Log a message with consistent formatting.

        Args:
            level: Logging level ('debug', 'info', 'warning', 'error', 'critical')
            message: Message to log
            exc_info: Whether to include exception information
        """
        log_message = f"{self.log_prefix}{message}"

        if level == "debug":
            logger.debug(log_message, exc_info=exc_info)
        elif level == "info":
            logger.info(log_message, exc_info=exc_info)
        elif level == "warning":
            logger.warning(log_message, exc_info=exc_info)
        elif level == "error":
            logger.error(log_message, exc_info=exc_info)
        elif level == "critical":
            logger.critical(log_message, exc_info=exc_info)

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
    ) -> Union[None, str]:
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
            error_msg = f"Executable not found: {exe}"
            self.log("error", error_msg)
            return f"Command execution failed: {error_msg}"

        self.log("info", f"Running command in venv: {cmd_str}")
        self.log(
            "debug", f"Command details: cwd={cwd}, capture_stdout={capture_stdout}"
        )

        # Save command to file for debugging
        if self.verbose_logging and args and args[0].endswith(".py"):
            try:
                script_path = os.path.join(cwd or ".", args[0])
                if os.path.exists(script_path):
                    with open(script_path, "r") as f:
                        script_content = f.read()
                    self.log(
                        "debug", f"Script content for {args[0]}:\n{script_content}"
                    )
            except Exception as e:
                self.log("debug", f"Could not read script file: {str(e)}")

        try:
            proc = subprocess.run(
                real_cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=300,  # 5-minute timeout to prevent hangs
            )

            # Always log detailed info about the execution
            self.log("debug", f"Command returned with code {proc.returncode}")

            # Log stdout and stderr regardless of return code
            if proc.stdout and self.verbose_logging:
                self.log("debug", f"STDOUT ({len(proc.stdout)} chars):\n{proc.stdout}")
            if proc.stderr and self.verbose_logging:
                self.log("debug", f"STDERR ({len(proc.stderr)} chars):\n{proc.stderr}")

            if proc.returncode != 0:
                error_output = f"Command failed with code {proc.returncode}\n"
                if proc.stdout:
                    error_output += f"STDOUT:\n{proc.stdout}\n"
                if proc.stderr:
                    error_output += f"STDERR:\n{proc.stderr}"

                self.log("error", f"Command failed: {cmd_str} (code={proc.returncode})")
                self.log("error", f"Full error output:\n{error_output}")

                return error_output.strip()

            # Return stdout if requested
            if capture_stdout and proc.stdout:
                self.log("debug", f"Command captured stdout ({len(proc.stdout)} chars)")
                return proc.stdout.strip()

        except subprocess.TimeoutExpired:
            error_msg = f"Command timed out after 300 seconds: {cmd_str}"
            self.log("error", error_msg)
            return f"Command execution failed: {error_msg}"
        except Exception as e:
            error_msg = f"Exception executing command: {str(e)}"
            self.log("error", error_msg, exc_info=True)
            return f"Command execution failed with exception: {str(e)}\n{traceback.format_exc()}"

        self.log("info", f"Command executed successfully: {cmd_str}")
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
        deps = parse_imports(code_str, is_cython)
        self.log("info", f"Parsed dependencies: {deps}")
        return deps

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
            self.log("debug", "Identified code as Cython")
        return is_cython

    def save_file_content(self, filepath: str, label: str = "file"):
        """
        Save the content of a file to the logs for debugging purposes.

        Args:
            filepath: Path to the file to save
            label: Label to use in the log message
        """
        if not self.verbose_logging:
            return

        try:
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    content = f.read()
                self.log("debug", f"Content of {label} at {filepath}:\n{content}")
            else:
                self.log("debug", f"{label.capitalize()} not found at {filepath}")
        except Exception as e:
            self.log("debug", f"Error reading {label} at {filepath}: {str(e)}")

    def setup_persistent_logs(self, log_dir: str = None):
        """
        Set up persistent logging to a file.

        Args:
            log_dir: Directory to store logs (default: current working directory)
        """
        if log_dir is None:
            log_dir = os.getcwd()

        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"build_{self.request_id}.log")

        # Create a file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Create a formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(file_handler)

        self.log("info", f"Persistent logging enabled to {log_file}")
        return log_file
