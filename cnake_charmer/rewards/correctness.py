"""
Correctness reward: fraction of test cases passing.

Compares Cython output against the Python reference for each test case.
"""

from collections.abc import Callable

from cnake_charmer.validate.pipeline import validate


def correctness_reward(
    cython_code: str,
    python_func: Callable,
    func_name: str,
    test_cases: list,
    **kwargs,
) -> float:
    """
    Return correctness score (0.0 to 1.0) based on test case pass rate.

    Only evaluates correctness — skips benchmarking for speed.
    """
    result = validate(
        cython_code=cython_code,
        python_func=python_func,
        func_name=func_name,
        test_cases=test_cases,
        skip_benchmark=True,
    )

    if not result.compiled:
        return 0.0

    if result.correctness is not None:
        return result.correctness.score

    return 0.0
