"""
Cython reward functions for code quality evaluation.

This module provides scoring functions for evaluating Cython code optimization quality,
integrating with the general reward system framework.
"""

import logging
import re
from typing import Dict, Any, Optional, List, Mapping

from .parsing.static_parser import is_cython_code
from .common import OptimizationScoreComponents, OptimizationHint, CythonAnalysisResult
from .reporting import metrics_to_analysis_result

# Configure logger
logger = logging.getLogger("cython_analyzer.reward")


def cython_analyzer_score(inputs: Dict[str, Any], outputs: Dict[str, Any]) -> float:
    """
    Calculate a score based on Cython code optimization quality using the Cython analyzer.

    Args:
        inputs: The inputs to the code generator
        outputs: The outputs from the code generator

    Returns:
        float: Cython optimization score between 0 and 1
    """
    # Import here to avoid circular import
    from .analyzer import CythonAnalyzer
    from cnake_charmer.generate.ephemeral_runner.builders.cython import CythonBuilder

    code = outputs.get("generated_code", "")
    if not code:
        logger.warning("No code found for Cython analyzer scoring")
        return 0.0

    # Check if this is actually Cython code
    if not is_cython_code(code):
        logger.info("Code does not contain Cython features, skipping analyzer scoring")
        return 1.0  # Not applicable

    try:
        # Create a CythonBuilder for analysis
        request_id = f"analyzer_score_{hash(code) % 10000}"
        builder = CythonBuilder(request_id=request_id)

        # Create a CythonAnalyzer instance
        analyzer = CythonAnalyzer(ephemeral_runner=builder)

        # Analyze the code
        logger.info("Running Cython code analysis")
        metrics = analyzer.analyze_code(code)

        # Get the optimization score from metrics
        optimization_score = metrics.get("optimization_score", 0.0)

        # Convert metrics to structured analysis result
        analysis_result = metrics_to_analysis_result(metrics)

        # Log detailed metrics
        logger.info(
            f"Cython analyzer score details: optimization_score={optimization_score:.2f}, "
            f"c_ratio={analysis_result.c_ratio:.2f}, python_ratio={analysis_result.python_ratio:.2f}"
        )

        # Store the structured analysis in outputs for potential use in feedback
        outputs["cython_analysis"] = analysis_result

        return optimization_score

    except Exception as e:
        logger.error(f"Error during Cython analysis: {str(e)}")
        error_result = CythonAnalysisResult(
            optimization_score=0.5,  # Neutral score on error
            success=False,
            error=str(e),
        )
        outputs["cython_analysis"] = error_result
        return 0.5  # Neutral score on error


def cython_usage_score(inputs: Dict[str, Any], outputs: Dict[str, Any]) -> float:
    """
    Calculate a score based on Cython feature usage without requiring compilation.
    This is a lighter alternative to cython_analyzer_score.

    Args:
        inputs: The inputs to the code generator
        outputs: The outputs from the code generator

    Returns:
        float: Cython usage score between 0 and 1
    """
    code = outputs.get("generated_code", "")
    if not code:
        logger.warning("No code found for Cython usage scoring")
        return 0.0

    # Check if Cython is expected based on the prompt
    prompt = inputs.get("prompt", "").lower()
    cython_expected = any(
        keyword in prompt
        for keyword in [
            "cython",
            "fast",
            "performance",
            "efficient",
            "speed",
            "optimize",
        ]
    )

    # If Cython is not expected, return full score (N/A)
    if not cython_expected:
        logger.info("Cython not expected based on prompt, skipping Cython scoring")
        return 1.0

    code_lines = code.split("\n")

    # Check if it's actually Cython code
    is_cython_code_result = is_cython_code(code)
    if not is_cython_code_result:
        logger.warning("Cython was expected but code does not contain Cython features")
        return 0.0  # Cython was expected but not delivered

    # Count Cython features
    features = {
        "cdef_vars": sum(
            1 for line in code_lines if re.search(r"cdef\s+(?!class)", line)
        ),
        "cdef_class": sum(1 for line in code_lines if "cdef class" in line),
        "cpdef_funcs": sum(1 for line in code_lines if "cpdef" in line),
        "memoryviews": sum(
            1 for line in code_lines if "[:]" in line or re.search(r"[a-z]+\[:", line)
        ),
        "c_types": sum(
            1
            for line in code_lines
            if any(
                t in line for t in ["int ", "float ", "double ", "bint ", "unsigned "]
            )
        ),
        "nogil": sum(1 for line in code_lines if "nogil" in line),
        "directives": sum(1 for line in code_lines if "# cython:" in line),
    }

    # Log detailed feature counts
    feature_details = ", ".join([f"{name}={count}" for name, count in features.items()])
    logger.info(f"Cython features detected: {feature_details}")

    # Calculate feature density
    total_features = sum(features.values())
    feature_density = total_features / max(1, len(code_lines) - features["directives"])
    logger.info(f"Cython feature density: {feature_density:.3f} features per line")

    # Base score from density
    if feature_density >= 0.4:  # At least 40% of lines have Cython features
        density_score = 1.0
    elif feature_density >= 0.2:  # At least 20% of lines have Cython features
        density_score = 0.5
    else:
        density_score = 0.2

    # Bonus for specific features
    feature_score = min(
        1.0,
        (
            (0.2 if features["cdef_vars"] > 0 else 0.0)
            + (0.2 if features["memoryviews"] > 0 else 0.0)
            + (0.2 if features["c_types"] > 2 else 0.0)
            + (0.2 if features["nogil"] > 0 else 0.0)
            + (0.2 if features["directives"] > 1 else 0.0)
        ),
    )

    # Combine scores
    final_cython_score = (density_score * 0.6) + (feature_score * 0.4)

    logger.info(
        f"Cython usage score details: density_score={density_score:.2f} ({feature_density:.3f}), "
        f"feature_score={feature_score:.2f}"
    )

    return final_cython_score


def create_cython_reward_system(base_system=None):
    """
    Create a reward system configured for Cython code evaluation.

    Args:
        base_system: Optional existing reward system to extend (if None, creates a new one)

    Returns:
        RewardSystem: Configured reward system with Cython scoring functions
    """
    # Import reward system here to avoid circular imports
    from reward_system import RewardSystem, create_default_reward_system

    # Create or use provided reward system
    reward_system = base_system or create_default_reward_system()

    # Register the analyzer-based scoring function
    reward_system.register_scoring_function(
        cython_analyzer_score, weight=0.7, name="cython_optimization"
    )

    # Optionally add the lighter usage score
    # If you want to use both, uncomment this:
    # reward_system.register_scoring_function(
    #     cython_usage_score, weight=0.3, name="cython_features"
    # )

    return reward_system
