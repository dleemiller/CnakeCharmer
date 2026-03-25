"""
Benchmark a Cython implementation against a Python reference.

Measures execution time of both and computes the speedup ratio.
"""

import logging
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    success: bool
    speedup: float = 0.0
    python_time: float = 0.0
    cython_time: float = 0.0
    python_std: float = 0.0
    cython_std: float = 0.0
    num_runs: int = 0
    error: str = ""


def run_benchmark(
    python_func: Callable,
    cython_func: Callable,
    args: tuple = (),
    kwargs: dict | None = None,
    num_runs: int = 10,
    warmup_runs: int = 2,
) -> BenchmarkResult:
    """
    Benchmark Python vs Cython implementations.

    Args:
        python_func: Python reference callable.
        cython_func: Cython callable to benchmark.
        args: Positional arguments to pass to both functions.
        kwargs: Keyword arguments to pass to both functions.
        num_runs: Number of timed runs.
        warmup_runs: Number of warmup runs before timing.

    Returns:
        BenchmarkResult with timing and speedup.
    """
    if kwargs is None:
        kwargs = {}

    result = BenchmarkResult(success=False, num_runs=num_runs)

    # Warmup
    try:
        for _ in range(warmup_runs):
            python_func(*args, **kwargs)
            cython_func(*args, **kwargs)
    except Exception as e:
        result.error = f"Warmup failed: {e}"
        return result

    # Time Python
    try:
        py_times = _time_function(python_func, args, kwargs, num_runs)
    except Exception as e:
        result.error = f"Python benchmark failed: {e}"
        return result

    # Time Cython
    try:
        cy_times = _time_function(cython_func, args, kwargs, num_runs)
    except Exception as e:
        result.error = f"Cython benchmark failed: {e}"
        return result

    result.python_time = statistics.mean(py_times)
    result.cython_time = statistics.mean(cy_times)
    result.python_std = statistics.stdev(py_times) if len(py_times) > 1 else 0.0
    result.cython_std = statistics.stdev(cy_times) if len(cy_times) > 1 else 0.0

    if result.cython_time > 0:
        result.speedup = result.python_time / result.cython_time
    else:
        result.speedup = float("inf")

    result.success = True
    logger.debug(
        f"Benchmark: Python={result.python_time:.6f}s, "
        f"Cython={result.cython_time:.6f}s, "
        f"Speedup={result.speedup:.2f}x"
    )
    return result


def _time_function(func: Callable, args: tuple, kwargs: dict, num_runs: int) -> list:
    """Time a function over multiple runs, return list of times."""
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return times
