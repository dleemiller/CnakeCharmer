import dspy
import logging
from typing import Dict

logger = logging.getLogger("cnake_charmer.feedback")

class FeedbackLoop:
    """Handles code regeneration based on test failures."""

    def __init__(self, model=None):
        self.model = model or dspy.LM(model="openrouter/anthropic/claude-3.7-sonnet", cache=False, max_tokens=2500)

    def refine_code(self, failed_logs: str, previous_code: Dict[str, str]) -> Dict[str, str]:
        """
        Generates a refined version of Python and Cython code based on failure logs.

        Args:
            failed_logs (str): Logs explaining the failure.
            previous_code (Dict[str, str]): The previous versions of Python and Cython code.

        Returns:
            Dict[str, str]: New versions of Python and Cython code.
        """
        task = dspy.Predict(
            name="RefineGeneratedCode",
            input_keys=["failed_logs", "previous_code"],
            output_keys=["new_python_code", "new_cython_code"]
        )

        response = self.model(task)(
            failed_logs=failed_logs,
            previous_code=previous_code
        )

        return {
            "python": response["new_python_code"],
            "cython": response["new_cython_code"]
        }