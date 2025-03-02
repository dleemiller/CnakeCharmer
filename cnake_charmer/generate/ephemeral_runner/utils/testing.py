"""
Testing utilities for ephemeral code.

This module provides functions for testing code in ephemeral environments.
"""

import os
import logging
import subprocess
from typing import Optional, Dict, Any

# Configure logger
logger = logging.getLogger("ephemeral_runner.utils.testing")


def create_test_script(module_name: str, tmpdir: str) -> str:
    """
    Create a test script that will import and test the module.

    Args:
        module_name: Name of the module to test
        tmpdir: Path to the directory where the test script should be created

    Returns:
        Path to the created test script
    """
    from ephemeral_runner.utils.templates import load_template

    test_script_path = os.path.join(tmpdir, "test_script.py")

    # Load the test script template and format it with the module name
    template = load_template("generic_test.py.template")
    test_script_content = template.format(module_name=module_name)

    with open(test_script_path, "w") as f:
        f.write(test_script_content)

    return test_script_path


def run_test_script(
    python_path: str, test_script_path: str, cwd: str = None
) -> Optional[str]:
    """
    Run a test script.

    Args:
        python_path: Path to the Python executable
        test_script_path: Path to the test script
        cwd: Working directory for the script

    Returns:
        Error message if failed, None if successful
    """
    try:
        proc = subprocess.run(
            [python_path, test_script_path], cwd=cwd, capture_output=True, text=True
        )

        if proc.returncode != 0:
            error_output = f"Test failed with code {proc.returncode}\n"
            if proc.stdout:
                error_output += f"STDOUT:\n{proc.stdout}\n"
            if proc.stderr:
                error_output += f"STDERR:\n{proc.stderr}"
            logger.error(f"Test script failed: {error_output}")
            return error_output.strip()

        return None
    except Exception as e:
        logger.error(f"Exception running test script: {str(e)}")
        return f"Test script execution failed with exception: {str(e)}"
