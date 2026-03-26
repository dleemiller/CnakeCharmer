"""
Performance reward: based on speedup over the Python reference.

Rewards faster Cython implementations with diminishing returns.
"""

import math
from collections.abc import Callable

from cnake_charmer.validate.pipeline import validate


def performance_reward(
    cython_code: str,
    python_func: Callable,
    func_name: str,
    test_cases: list | None = None,
    benchmark_args: tuple | None = None,
    benchmark_runs: int = 5,
    max_reward: float = 1.0,
    **kwargs,
) -> float:
    """
    Return a reward based on the speedup achieved.

    Reward curve (log-scaled, diminishing returns):
    - Slower than Python: 0.0
    - 1x (same speed): 0.0
    - 2x speedup: ~0.3
    - 5x speedup: ~0.7
    - 10x+ speedup: ~1.0

    Args:
        max_reward: Cap on the reward value.
    """
    result = validate(
        cython_code=cython_code,
        python_func=python_func,
        func_name=func_name,
        test_cases=test_cases,
        benchmark_args=benchmark_args,
        benchmark_runs=benchmark_runs,
        skip_correctness=True,
    )

    if not result.compiled or result.benchmark is None or not result.benchmark.success:
        return 0.0

    speedup = result.benchmark.speedup
    if speedup <= 1.0:
        return 0.0

    # Log-scaled reward: log2(speedup) / log2(10), capped at max_reward
    # 2x -> 0.3, 5x -> 0.7, 10x -> 1.0
    reward = math.log2(speedup) / math.log2(10)
    return min(reward, max_reward)
