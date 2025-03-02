import logging

logger = logging.getLogger("cython_analyzer.metrics")


def calculate_optimization_scores(metrics):
    """
    Calculate optimization scores based on analysis metrics.

    Args:
        metrics: Analysis metrics from static analysis and HTML parsing

    Returns:
        dict: Input metrics augmented with optimization scores
    """
    # Make a copy to avoid modifying the input
    result = metrics.copy()

    # Calculate derived metrics
    if "total_lines" in result and result["total_lines"] > 0:
        result["c_ratio"] = result.get("c_lines", 0) / result["total_lines"]
        result["python_ratio"] = result.get("python_lines", 0) / result["total_lines"]
    else:
        result["c_ratio"] = result.get("static_c_ratio", 0)
        result["python_ratio"] = 1.0 - result["c_ratio"]

    # Calculate the optimization score
    _calculate_optimization_score(result)

    return result


def _calculate_optimization_score(metrics):
    """
    Calculate the overall optimization score based on analysis metrics.

    Args:
        metrics: The metrics dictionary to update with the optimization score
    """
    # Set defaults for missing metrics
    c_ratio = metrics.get("c_ratio", metrics.get("static_c_ratio", 0))
    python_ratio = metrics.get("python_ratio", 1.0 - c_ratio)
    gil_ops = metrics.get("gil_operations", 0)
    vec_loops = metrics.get("vectorizable_loops", 0)
    unopt_math = metrics.get("unoptimized_math", 0)
    total_lines = metrics.get("total_lines", 1)

    # Calculate feature quality score from code features
    code_features = metrics.get("code_features", {})
    feature_score = 0.0

    # Award points for each optimization feature
    if code_features.get("nogil", 0) > 0:
        feature_score += 0.2

    if code_features.get("prange", 0) > 0:
        feature_score += 0.2

    if code_features.get("memoryviews", 0) > 0:
        feature_score += 0.2

    if code_features.get("directives", 0) > 0:
        feature_score += 0.1

    if code_features.get("cdef_vars", 0) / max(1, total_lines) > 0.1:
        feature_score += 0.2

    if code_features.get("cdef_funcs", 0) + code_features.get("cpdef_funcs", 0) > 0:
        feature_score += 0.1

    # Normalize feature score to 0-1 range
    feature_score = min(1.0, feature_score)

    # Overall optimization score (weighted components)
    optimization_score = (
        # Reward high C ratio (40% weight)
        (c_ratio * 0.4)
        +
        # Penalize Python interaction (20% weight)
        ((1.0 - python_ratio) * 0.2)
        +
        # Penalize GIL operations (10% weight)
        ((1.0 - min(1.0, gil_ops / max(1, total_lines))) * 0.1)
        +
        # Penalize unvectorized loops (10% weight)
        ((1.0 - min(1.0, vec_loops / max(1, total_lines))) * 0.1)
        +
        # Reward good Cython features (20% weight)
        (feature_score * 0.2)
    )

    # Store all component scores for detailed reporting
    metrics["component_scores"] = {
        "c_ratio_score": c_ratio,
        "python_interaction_score": 1.0 - python_ratio,
        "gil_operations_score": 1.0 - min(1.0, gil_ops / max(1, total_lines)),
        "vectorizable_loops_score": 1.0 - min(1.0, vec_loops / max(1, total_lines)),
        "feature_score": feature_score,
    }

    metrics["optimization_score"] = optimization_score
    logger.info(f"Calculated optimization score: {optimization_score:.2f}")
