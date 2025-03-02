import subprocess
import logging
from typing import Dict, Optional

logger = logging.getLogger("cnake_charmer.equivalency")

class EquivalencyChecker:
    """Runs equivalency tests on Python and Cython implementations."""

    def __init__(self, python_path: str, cython_path: str):
        self.python_path = python_path
        self.cython_path = cython_path

    def run_test(self, executable: str, test_script: str) -> Optional[str]:
        """Runs a test script and returns errors if any."""
        try:
            proc = subprocess.run([executable, test_script], capture_output=True, text=True)
            if proc.returncode != 0:
                return proc.stderr
            return None
        except Exception as e:
            return f"Execution failed: {str(e)}"

    def check_equivalency(self) -> bool:
        """Runs Python and Cython implementations and checks if outputs match."""
        python_result = self.run_test(self.python_path, "test_script.py")
        cython_result = self.run_test(self.cython_path, "test_script.py")

        if python_result or cython_result:
            logger.error(f"Equivalency Test Failed: {python_result} || {cython_result}")
            return False
        return True