"""
Virtual environment management utilities.

This module provides functions for creating and managing virtual environments.
"""

import os
import sys
import venv
import logging
import subprocess
from typing import List, Optional, Dict, Any

from ephemeral_runner.exceptions import VenvCreationError

# Configure logger
logger = logging.getLogger("ephemeral_runner.utils.venv")


def create_venv(venv_path: str, with_pip: bool = True) -> bool:
    """
    Create a virtual environment.

    Args:
        venv_path: Path where the virtual environment should be created
        with_pip: Whether to include pip in the environment

    Returns:
        True if successful

    Raises:
        VenvCreationError: If virtual environment creation fails
    """
    logger.info(f"Creating virtual environment at {venv_path}")
    try:
        venv.create(venv_path, with_pip=with_pip)
        logger.debug(f"Successfully created virtual environment at {venv_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create virtual environment: {str(e)}")
        raise VenvCreationError(f"Failed to create virtual environment: {str(e)}")


def get_venv_python_path(venv_path: str) -> str:
    """
    Get the path to the Python executable in a virtual environment.

    Args:
        venv_path: Path to the virtual environment

    Returns:
        Path to the Python executable
    """
    if sys.platform == "win32":
        return os.path.join(venv_path, "Scripts", "python.exe")
    else:
        return os.path.join(venv_path, "bin", "python")


def get_venv_pip_path(venv_path: str) -> str:
    """
    Get the path to the pip executable in a virtual environment.

    Args:
        venv_path: Path to the virtual environment

    Returns:
        Path to the pip executable
    """
    if sys.platform == "win32":
        return os.path.join(venv_path, "Scripts", "pip.exe")
    else:
        return os.path.join(venv_path, "bin", "pip")


def install_dependencies(
    venv_path: str, dependencies: List[str], upgrade_pip: bool = True
) -> Optional[str]:
    """
    Install dependencies in a virtual environment.

    Args:
        venv_path: Path to the virtual environment
        dependencies: List of dependencies to install
        upgrade_pip: Whether to upgrade pip first

    Returns:
        Error message if failed, None if successful
    """
    if not dependencies and not upgrade_pip:
        return None

    pip_path = get_venv_pip_path(venv_path)

    if upgrade_pip:
        logger.info(f"Upgrading pip in {venv_path}")
        upgrade_cmd = [pip_path, "install", "--upgrade", "pip", "wheel", "setuptools"]
        try:
            proc = subprocess.run(upgrade_cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                error_msg = f"Failed to upgrade pip: {proc.stderr}"
                logger.error(error_msg)
                return error_msg
        except Exception as e:
            error_msg = f"Failed to upgrade pip: {str(e)}"
            logger.error(error_msg)
            return error_msg

    if dependencies:
        logger.info(f"Installing dependencies in {venv_path}: {dependencies}")
        install_cmd = [pip_path, "install"] + dependencies
        try:
            proc = subprocess.run(install_cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                error_msg = f"Failed to install dependencies: {proc.stderr}"
                logger.error(error_msg)
                return error_msg
        except Exception as e:
            error_msg = f"Failed to install dependencies: {str(e)}"
            logger.error(error_msg)
            return error_msg

    return None
