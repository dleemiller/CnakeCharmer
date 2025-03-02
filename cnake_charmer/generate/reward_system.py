"""
Reward System Framework

This module provides a flexible, composable reward system for evaluating code quality.
It allows combining multiple scoring functions with different weights.
"""

import logging
import re
from typing import Dict, Any, List, Callable, Tuple, Optional, Union
import traceback
import json

# Configure logger
logger = logging.getLogger("reward_system")

# Type definitions
ScoringFunction = Callable[[Dict[str, Any], Dict[str, Any]], float]
WeightedScoringFunction = Tuple[ScoringFunction, float, str]  # function, weight, name


class RewardSystem:
    """
    A flexible reward system that combines multiple scoring functions.

    This system allows for modular evaluation of code quality by:
    1. Registering multiple scoring functions
    2. Assigning weights to each function
    3. Combining scores with transparent reporting
    """

    def __init__(self, base_error_score: float = -1.0):
        """
        Initialize the reward system.

        Args:
            base_error_score: The score to return when critical errors occur
        """
        self.scoring_functions: List[WeightedScoringFunction] = []
        self.base_error_score = base_error_score
        self.last_detailed_scores = {}  # Store the last detailed scores for reporting

    def register_scoring_function(
        self, function: ScoringFunction, weight: float = 1.0, name: Optional[str] = None
    ) -> None:
        """
        Register a scoring function with the reward system.

        Args:
            function: The scoring function to register
            weight: The weight to apply to this function's score (default: 1.0)
            name: Optional name for the function (for logging)
        """
        func_name = name or function.__name__
        self.scoring_functions.append((function, weight, func_name))
        logger.info(f"Registered scoring function '{func_name}' with weight {weight}")

    def calculate_reward(
        self, inputs: Dict[str, Any], outputs: Dict[str, Any]
    ) -> float:
        """
        Calculate the combined reward using all registered scoring functions.

        Args:
            inputs: The inputs provided to the code generator
            outputs: The outputs from the code generator

        Returns:
            float: The combined reward score
        """
        # Clear previous detailed scores
        self.last_detailed_scores = {
            "individual_scores": {},
            "weighted_scores": {},
            "total_score": 0.0,
            "error": None,
        }

        # Check for critical errors first
        if outputs.get("error"):
            error_score, error_message = self._handle_error(outputs["error"])
            if error_score is not None:
                self.last_detailed_scores["error"] = error_message
                self.last_detailed_scores["total_score"] = error_score
                return error_score

        # Check if execution failed
        if "execution" in outputs and not outputs["execution"].get("success", True):
            error_message = outputs["execution"].get(
                "stderr", "Unknown execution error"
            )
            error_score = -0.5  # Default penalty for execution failure
            logger.warning(f"Execution failed: {error_message[:200]}...")
            self.last_detailed_scores["error"] = (
                f"Execution failed: {error_message[:200]}..."
            )
            self.last_detailed_scores["total_score"] = error_score
            return error_score

        # If no critical errors, calculate the combined score
        total_score = 0.0
        total_weight = 0.0
        scores = {}
        weighted_scores = {}

        # Run each scoring function and collect results
        for func, weight, name in self.scoring_functions:
            try:
                # Call the scoring function
                score = func(inputs, outputs)

                # Store the raw and weighted scores
                scores[name] = score
                weighted_score = score * weight
                weighted_scores[name] = weighted_score

                # Add to totals
                total_score += weighted_score
                total_weight += weight

                logger.info(
                    f"Scoring function '{name}' returned {score:.2f} (weighted: {weighted_score:.2f})"
                )
            except Exception as e:
                logger.error(f"Error in scoring function '{name}': {str(e)}")
                logger.debug(traceback.format_exc())
                # Skip this function but continue with others

        # If no scoring functions succeeded, return base error score
        if total_weight == 0:
            logger.error("No scoring functions succeeded")
            self.last_detailed_scores["error"] = "All scoring functions failed"
            self.last_detailed_scores["total_score"] = self.base_error_score
            return self.base_error_score

        # Calculate the final score
        final_score = total_score / total_weight

        # Store detailed scores for reporting
        self.last_detailed_scores["individual_scores"] = scores
        self.last_detailed_scores["weighted_scores"] = weighted_scores
        self.last_detailed_scores["total_score"] = final_score
        self.last_detailed_scores["total_weight"] = total_weight

        # Log the detailed breakdown
        score_details = ", ".join(
            [
                f"{name}={score:.2f} (weighted: {weighted_scores[name]:.2f})"
                for name, score in scores.items()
            ]
        )
        logger.info(f"Reward breakdown: {score_details}")
        logger.info(f"Final reward: {final_score:.2f}")

        return final_score

    def _handle_error(self, error_text: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Handle error cases and assign appropriate scores.

        Args:
            error_text: The error message from the code generation/execution

        Returns:
            tuple: A score for the error and an error message, or (None, None) to continue with scoring functions
        """
        error_text = str(error_text).lower()

        # Map error types to scores and messages
        error_types = [
            ("syntax error", -1.0, "Syntax Error"),
            ("parse error", -1.0, "Parse Error"),
            ("import error", -0.8, "Import Error"),
            ("no module named", -0.8, "Module Not Found"),
            ("type error", -0.7, "Type Error"),
            ("attribute error", -0.7, "Attribute Error"),
            ("runtime error", -0.6, "Runtime Error"),
            ("exception", -0.6, "Exception"),
        ]

        # Check for known error types
        for error_pattern, score, message in error_types:
            if error_pattern in error_text:
                logger.info(f"Reward: {score:.1f} ({message})")
                return score, f"{message}: {error_text[:200]}..."

        # Generic error case
        logger.info(f"Reward: {self.base_error_score:.1f} (Unknown Error)")
        return self.base_error_score, f"Unknown Error: {error_text[:200]}..."

    def get_detailed_scores(self) -> Dict[str, Any]:
        """
        Get detailed breakdown of the last calculated scores.

        Returns:
            dict: Detailed score information
        """
        return self.last_detailed_scores

    def get_score_explanation(self) -> str:
        """
        Generate a human-readable explanation of the last calculated scores.

        Returns:
            str: Explanation of how the score was calculated
        """
        if not self.last_detailed_scores:
            return "No scores have been calculated yet."

        if "error" in self.last_detailed_scores and self.last_detailed_scores["error"]:
            return f"Error occurred: {self.last_detailed_scores['error']}\nFinal score: {self.last_detailed_scores['total_score']:.2f}"

        explanation = ["Score Breakdown:"]
        individual_scores = self.last_detailed_scores.get("individual_scores", {})
        weighted_scores = self.last_detailed_scores.get("weighted_scores", {})

        for name, score in individual_scores.items():
            weighted = weighted_scores.get(name, 0.0)
            weight = weighted / score if score != 0 else 0
            explanation.append(
                f"- {name}: {score:.2f} (weight: {weight:.2f}, weighted score: {weighted:.2f})"
            )

        explanation.append(
            f"\nFinal Score: {self.last_detailed_scores.get('total_score', 0.0):.2f}"
        )

        return "\n".join(explanation)


# Predefined scoring functions


def style_score(inputs: Dict[str, Any], outputs: Dict[str, Any]) -> float:
    """
    Calculate a score based on code style.

    Args:
        inputs: The inputs to the code generator
        outputs: The outputs from the code generator

    Returns:
        float: Style score between 0 and 1
    """
    code = outputs.get("generated_code", "")
    if not code:
        logger.warning("No code found for style scoring")
        return 0.0

    code_lines = code.split("\n")
    total_lines = len(code_lines)

    # Initialize sub-scores
    line_length_score = 0.0
    indentation_score = 0.0
    naming_score = 0.0

    # Check line length (PEP 8 recommends <= 79 chars)
    long_lines = sum(1 for line in code_lines if len(line) > 79)
    if long_lines == 0:
        line_length_score = 1.0
    else:
        line_length_score = max(0.0, 1.0 - (long_lines / total_lines))

    # Check for consistent indentation
    indentation_pattern = re.compile(r"^(\s*)\S")
    indentation_types = set()
    for line in code_lines:
        match = indentation_pattern.match(line)
        if match and match.group(1):
            indentation_types.add(match.group(1))

    # Fewer indentation types is better
    if (
        len(indentation_types) <= 2
    ):  # Allow for zero indentation and one indentation level
        indentation_score = 1.0
    else:
        indentation_score = max(0.0, 1.0 - ((len(indentation_types) - 2) * 0.25))

    # Check for proper function/variable naming (snake_case for Python)
    camel_case_pattern = re.compile(r"[a-z][a-z0-9]*[A-Z]")
    camel_case_names = sum(1 for line in code_lines if camel_case_pattern.search(line))

    if camel_case_names == 0:  # Proper snake_case throughout
        naming_score = 1.0
    else:
        naming_score = max(0.0, 1.0 - (camel_case_names / total_lines))

    # Combine sub-scores
    final_style_score = (line_length_score + indentation_score + naming_score) / 3.0

    logger.info(
        f"Style score details: line_length={line_length_score:.2f}, "
        f"indentation={indentation_score:.2f}, naming={naming_score:.2f}"
    )

    return final_style_score


def documentation_score(inputs: Dict[str, Any], outputs: Dict[str, Any]) -> float:
    """
    Calculate a score based on code documentation.

    Args:
        inputs: The inputs to the code generator
        outputs: The outputs from the code generator

    Returns:
        float: Documentation score between 0 and 1
    """
    code = outputs.get("generated_code", "")
    if not code:
        logger.warning("No code found for documentation scoring")
        return 0.0

    code_lines = code.split("\n")
    total_lines = len(code_lines)

    # Check for module-level docstring
    has_module_docstring = False
    for i, line in enumerate(code_lines[:10]):  # Check first 10 lines
        if '"""' in line or "'''" in line:
            has_module_docstring = True
            break

    # Count docstrings in functions and classes
    docstring_count = 0
    function_count = 0

    for i, line in enumerate(code_lines):
        # Match function or class definitions
        if re.match(r"^\s*(def|cdef|cpdef|class)", line):
            function_count += 1

            # Look for docstring in next lines
            for j in range(i + 1, min(i + 10, total_lines)):
                if '"""' in code_lines[j] or "'''" in code_lines[j]:
                    docstring_text = ""
                    # Find the end of the docstring
                    for k in range(j, min(j + 20, total_lines)):
                        docstring_text += code_lines[k]
                        if ('"""' in code_lines[k] or "'''" in code_lines[k]) and k > j:
                            break

                    # Check if it looks like a Google-style docstring
                    if any(
                        section in docstring_text
                        for section in [
                            "Args:",
                            "Returns:",
                            "Yields:",
                            "Raises:",
                            "Attributes:",
                            "Example:",
                        ]
                    ):
                        docstring_count += 1
                    else:
                        # Simple docstring (less points)
                        docstring_count += 0.5
                    break

    # Calculate docstring coverage
    docstring_coverage = (
        docstring_count / max(1, function_count) if function_count > 0 else 0
    )

    # Check for inline comments
    comment_lines = sum(
        1
        for line in code_lines
        if line.strip().startswith("#")
        and not any(directive in line for directive in ["cython:", "distutils:"])
    )
    comment_ratio = comment_lines / max(total_lines, 1)

    # Ideal comment ratio is between 10% and 25% of code lines
    comment_score = 0.0
    if 0.1 <= comment_ratio <= 0.25:
        comment_score = 1.0
    elif comment_ratio > 0 and comment_ratio < 0.1:
        comment_score = 0.5  # Some comments, but could use more
    elif comment_ratio > 0.25:
        comment_score = 0.3  # Too many comments

    # Combine scores
    final_doc_score = (
        (0.2 if has_module_docstring else 0.0)  # 20% for module docstring
        + (docstring_coverage * 0.6)  # 60% for function docstrings
        + (comment_score * 0.2)  # 20% for proper comment ratio
    )

    logger.info(
        f"Documentation score details: module_docstring={0.2 if has_module_docstring else 0.0}, "
        f"docstring_coverage={docstring_coverage:.2f} ({docstring_count}/{function_count}), "
        f"comment_score={comment_score:.2f} (ratio: {comment_ratio:.2f})"
    )

    return final_doc_score


def create_default_reward_system() -> RewardSystem:
    """
    Create a default reward system with standard scoring functions.

    Returns:
        RewardSystem: Configured reward system with basic style and documentation scoring
    """
    reward_system = RewardSystem()

    # Register only the general scoring functions
    reward_system.register_scoring_function(style_score, weight=0.2, name="style")

    reward_system.register_scoring_function(
        documentation_score, weight=0.3, name="documentation"
    )

    return reward_system
