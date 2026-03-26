"""
Composite reward: weighted combination of all reward signals.

Compilation acts as a gate — if code doesn't compile, total reward is 0.
"""

import logging
from collections.abc import Callable

from cnake_charmer.validate.pipeline import validate

logger = logging.getLogger(__name__)

# Default weights
DEFAULT_WEIGHTS = {
    "correctness": 0.4,
    "performance": 0.3,
    "annotations": 0.3,
}


def composite_reward(
    cython_code: str,
    python_func: Callable,
    func_name: str,
    test_cases: list,
    benchmark_args: tuple | None = None,
    benchmark_runs: int = 5,
    weights: dict | None = None,
    **kwargs,
) -> dict:
    """
    Compute the full composite reward.

    Returns a dict with individual scores and the weighted total.
    Compilation is a gate — 0 total if it fails.

    Returns:
        {
            "total": float,
            "compiled": bool,
            "correctness": float,
            "performance": float,
            "annotations": float,
            "speedup": float,
            "compilation_errors": str,
            "correctness_failures": list,
            "annotation_hints": list,
        }
    """
    w = weights or DEFAULT_WEIGHTS

    # Run the full validation pipeline once (avoids compiling multiple times)
    result = validate(
        cython_code=cython_code,
        python_func=python_func,
        func_name=func_name,
        test_cases=test_cases,
        benchmark_args=benchmark_args,
        benchmark_runs=benchmark_runs,
    )

    scores = {
        "compiled": result.compiled,
        "compilation_errors": result.compilation.errors if result.compilation else "",
        "correctness": 0.0,
        "performance": 0.0,
        "annotations": 0.0,
        "speedup": 0.0,
        "correctness_failures": [],
        "annotation_hints": [],
        "total": 0.0,
    }

    # Gate: must compile
    if not result.compiled:
        return scores

    # Correctness score
    if result.correctness is not None:
        scores["correctness"] = result.correctness.score
        scores["correctness_failures"] = result.correctness.failures

    # Performance score (log-scaled speedup)
    if result.benchmark is not None and result.benchmark.success:
        import math

        speedup = result.benchmark.speedup
        scores["speedup"] = speedup
        if speedup > 1.0:
            scores["performance"] = min(math.log2(speedup) / math.log2(10), 1.0)

    # Annotation score
    if result.annotations is not None and result.annotations.success:
        scores["annotations"] = result.annotations.score
        scores["annotation_hints"] = result.annotations.hints

    # Weighted total
    scores["total"] = (
        w.get("correctness", 0.4) * scores["correctness"]
        + w.get("performance", 0.3) * scores["performance"]
        + w.get("annotations", 0.3) * scores["annotations"]
    )

    return scores
