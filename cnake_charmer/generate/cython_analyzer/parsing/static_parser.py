"""
Static analyzer for Cython code without compilation.
"""

import re
import logging

logger = logging.getLogger("cython_analyzer.static")


def analyze_static_features(code_str):
    """
    Perform static analysis of Cython code without compilation.

    Args:
        code_str: The Cython code to analyze

    Returns:
        dict: Static analysis metrics
    """
    lines = code_str.strip().split("\n")
    metrics = {
        "total_lines": len(lines),
        "line_contents": {},  # Store the content of each line
        "code_features": {
            "cdef_vars": 0,  # Number of cdef variable declarations
            "cdef_funcs": 0,  # Number of cdef function declarations
            "cpdef_funcs": 0,  # Number of cpdef function declarations
            "memoryviews": 0,  # Number of memoryview type declarations
            "typed_args": 0,  # Number of typed function arguments
            "nogil": 0,  # Number of nogil blocks or functions
            "prange": 0,  # Number of parallel range loops
            "directives": 0,  # Number of cython directives
        },
        "static_c_ratio": 0.0,  # Estimated C-to-Python ratio from static analysis
    }

    # Store line contents for reference
    for i, line in enumerate(lines):
        metrics["line_contents"][i + 1] = line.strip()

    # Count features using regex patterns
    for i, line in enumerate(lines):
        stripped = line.strip()

        # Cython directives
        if "# cython:" in stripped:
            metrics["code_features"]["directives"] += 1

        # cdef variables
        if re.search(r"cdef\s+(?!class|(?:class\s+)?(?:struct|enum|union))", stripped):
            metrics["code_features"]["cdef_vars"] += 1

        # cdef functions
        if re.search(r"cdef\s+\w+[\s\w\[\],\*]*\s+\w+\s*\(", stripped):
            metrics["code_features"]["cdef_funcs"] += 1

        # cpdef functions
        if re.search(r"cpdef\s+\w+[\s\w\[\],\*]*\s+\w+\s*\(", stripped):
            metrics["code_features"]["cpdef_funcs"] += 1

        # Memoryviews
        if "[:]" in stripped or "[:,:]" in stripped:
            metrics["code_features"]["memoryviews"] += 1

        # Typed arguments (in function declarations)
        if re.search(r"\(\s*\w+\s*:\s*\w+", stripped) or re.search(
            r",\s*\w+\s*:\s*\w+", stripped
        ):
            metrics["code_features"]["typed_args"] += 1

        # nogil blocks or functions
        if "nogil" in stripped:
            metrics["code_features"]["nogil"] += 1

        # Parallel range (prange)
        if "prange" in stripped:
            metrics["code_features"]["prange"] += 1

    # Calculate feature density
    c_features_count = sum(metrics["code_features"].values())
    metrics["feature_density"] = c_features_count / max(1, len(lines))

    # Estimate C-to-Python ratio based on feature density
    if metrics["feature_density"] > 0.3:
        metrics["static_c_ratio"] = 0.7  # High density of Cython features
    elif metrics["feature_density"] > 0.1:
        metrics["static_c_ratio"] = 0.5  # Medium density
    else:
        metrics["static_c_ratio"] = 0.3  # Low density

    logger.info(
        f"Static analysis: {c_features_count} Cython features found, density: {metrics['feature_density']:.3f}"
    )
    return metrics


def is_cython_code(code_str):
    """
    Check if a code string appears to be Cython.

    Args:
        code_str: The code string to check

    Returns:
        bool: True if the code contains Cython-specific elements
    """
    cython_indicators = ["cdef", "cpdef", "cimport", "nogil", "# cython:"]
    return any(indicator in code_str for indicator in cython_indicators)
