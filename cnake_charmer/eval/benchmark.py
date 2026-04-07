"""
Benchmark a Cython implementation against a Python reference.

Measures execution time of both and computes the speedup ratio.
Supports both in-process (legacy) and sandboxed out-of-process execution.
"""

import json
import logging
import os
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
    python_func: Callable | None = None,
    cython_func: Callable | None = None,
    args: tuple = (),
    kwargs: dict | None = None,
    num_runs: int = 10,
    warmup_runs: int = 2,
    max_total_seconds: float = 5.0,
    # New parameters for sandboxed execution
    python_code: str | None = None,
    func_name: str | None = None,
    cython_module_path: str | None = None,
) -> BenchmarkResult:
    """
    Benchmark Python vs Cython implementations.

    For sandboxed execution (preferred), provide:
    - python_code + func_name + cython_module_path + args

    For legacy in-process execution (backward compat), provide:
    - python_func + cython_func + args

    Args:
        python_func: Python reference callable (legacy, in-process).
        cython_func: Cython callable to benchmark (legacy, in-process).
        python_code: Python source code string (sandboxed).
        func_name: Function name to benchmark (sandboxed).
        cython_module_path: Path to compiled .so file (sandboxed).
        args: Positional arguments to pass to both functions.
        kwargs: Keyword arguments to pass to both functions.
        num_runs: Number of timed runs.
        warmup_runs: Number of warmup runs before timing.
        max_total_seconds: Maximum total wall time per function (warmup + timing).

    Returns:
        BenchmarkResult with timing and speedup.
    """
    if kwargs is None:
        kwargs = {}

    # Sandboxed path
    if python_code and func_name and cython_module_path:
        return _run_benchmark_sandboxed(
            python_code=python_code,
            func_name=func_name,
            cython_module_path=cython_module_path,
            args=args,
            kwargs=kwargs,
            num_runs=num_runs,
            warmup_runs=warmup_runs,
            max_total_seconds=max_total_seconds,
        )

    result = BenchmarkResult(success=False, num_runs=num_runs)

    # Timed warmup — bail early if a single call is too slow
    try:
        for _i in range(warmup_runs):
            start = time.perf_counter()
            python_func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            if elapsed > max_total_seconds:
                logger.debug(f"Warmup: Python call took {elapsed:.2f}s, reducing num_runs")
                num_runs = min(num_runs, 2)
            cython_func(*args, **kwargs)
    except Exception as e:
        result.error = f"Warmup failed: {e}"
        return result

    # Time Python
    try:
        py_times = _time_function(python_func, args, kwargs, num_runs, max_total_seconds)
    except Exception as e:
        result.error = f"Python benchmark failed: {e}"
        return result

    # Time Cython
    try:
        cy_times = _time_function(cython_func, args, kwargs, num_runs, max_total_seconds)
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


def _time_function(
    func: Callable, args: tuple, kwargs: dict, num_runs: int, max_total_seconds: float = 5.0
) -> list:
    """Time a function over multiple runs, return list of times.

    Stops early if cumulative time exceeds max_total_seconds, ensuring
    benchmarks don't hang on slow problems.
    """
    times = []
    cumulative = 0.0
    for _ in range(num_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        cumulative += elapsed
        if cumulative > max_total_seconds and len(times) >= 2:
            logger.debug(f"Benchmark cutoff after {len(times)} runs ({cumulative:.2f}s)")
            break
    return times


# ---------------------------------------------------------------------------
# Sandboxed benchmark
# ---------------------------------------------------------------------------


def _run_benchmark_sandboxed(
    python_code: str,
    func_name: str,
    cython_module_path: str,
    args: tuple = (),
    kwargs: dict | None = None,
    num_runs: int = 10,
    warmup_runs: int = 2,
    max_total_seconds: float = 5.0,
) -> BenchmarkResult:
    """Run benchmark in a sandboxed subprocess."""
    from cnake_charmer.eval.sandbox import execute_config, run_runner_sandboxed

    module_dir = os.path.dirname(os.path.abspath(cython_module_path))

    # Wall clock: enough for warmup + timing of both functions
    total_timeout = max(30, int(max_total_seconds * 3) + 15)
    sandbox_cfg = execute_config(
        wall_time_limit_s=total_timeout,
        cpu_time_limit_s=total_timeout - 5,
        extra_ro_binds=(module_dir,),
    )

    result = run_runner_sandboxed(
        "benchmark_runner",
        {
            "python_code": python_code,
            "func_name": func_name,
            "cython_module_path": cython_module_path,
            "args": list(args),
            "kwargs": kwargs or {},
            "num_runs": num_runs,
            "warmup_runs": warmup_runs,
            "max_total_seconds": max_total_seconds,
        },
        config=sandbox_cfg,
    )

    if result.timed_out:
        return BenchmarkResult(
            success=False,
            error=f"Benchmark timed out after {total_timeout}s",
        )

    if result.returncode != 0:
        return BenchmarkResult(
            success=False,
            error=f"Benchmark runner failed (rc={result.returncode}): {result.stderr[:500]}",
        )

    try:
        data = json.loads(result.stdout.strip())
    except (json.JSONDecodeError, ValueError):
        return BenchmarkResult(
            success=False,
            error=f"Failed to parse benchmark output: {result.stdout[:500]}",
        )

    if "error" in data:
        return BenchmarkResult(success=False, error=data["error"])

    return BenchmarkResult(
        success=data.get("success", True),
        speedup=data.get("speedup", 0.0),
        python_time=data.get("python_time", 0.0),
        cython_time=data.get("cython_time", 0.0),
        python_std=data.get("python_std", 0.0),
        cython_std=data.get("cython_std", 0.0),
        num_runs=data.get("num_runs", 0),
    )
