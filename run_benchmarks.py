#!/usr/bin/env python
"""
Benchmark Runner for Python vs. Cython Implementations.

This script dynamically imports all submodules under 'cnake_charmer' so that any
benchmark decorators register their functions into the global registry. It then
discovers benchmark pairs, executes each pair (after verifying that input parameters
match), computes timing statistics and speedup, and writes a Markdown report summarizing
all benchmarks.

Each row in the report corresponds to one benchmark pair.
"""

import time
import statistics
import logging
import pkgutil
import importlib
from typing import Tuple, Dict, Any, List

from cnake_charmer.benchmarks.registry import benchmark_registry, Variant, BenchmarkItem

# Configure logging.
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def import_all_submodules(package_name: str) -> None:
    """Dynamically import all submodules of a package.

    This ensures that all modules are loaded so that any decorators (e.g. benchmark
    registrations) are executed and the global registry is fully populated.

    Args:
        package_name (str): The package name to import, e.g., "cnake_charmer".
    """
    package = importlib.import_module(package_name)
    for loader, module_name, is_pkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        try:
            importlib.import_module(module_name)
        except Exception as e:
            logging.warning(f"Could not import module {module_name}: {e}")


def run_benchmark(item: BenchmarkItem) -> Tuple[float, float]:
    """Run a benchmark implementation repeatedly.

    Args:
        item (BenchmarkItem): The benchmark item containing the function and its call parameters.

    Returns:
        Tuple[float, float]: A tuple containing the average execution time and the standard
        deviation (in seconds).
    """
    func = item.func
    args = item.args
    kwargs = item.kwargs
    num_runs = item.num_runs

    # Warm-up run.
    func(*args, **kwargs)
    times: List[float] = []
    for _ in range(num_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        times.append(time.perf_counter() - start)
    avg = sum(times) / num_runs
    std = statistics.stdev(times) if num_runs > 1 else 0.0
    return avg, std


def run_all_benchmarks() -> List[Dict[str, Any]]:
    """Iterate over all benchmark pairs in the registry and run them.

    Only benchmark pairs where both the Python and Cython variants are registered
    are processed. Furthermore, if the two variants do not share identical input
    parameters, that benchmark pair is skipped with a warning.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing benchmark results.
        Each dictionary contains:
          - 'benchmark': Benchmark ID.
          - 'py_avg': Average time for the Python implementation.
          - 'py_std': Standard deviation for the Python implementation.
          - 'cy_avg': Average time for the Cython implementation.
          - 'cy_std': Standard deviation for the Cython implementation.
          - 'speedup': Speedup factor (py_avg / cy_avg).
    """
    results: List[Dict[str, Any]] = []
    for benchmark_id, variants in benchmark_registry.items():
        if Variant.PYTHON not in variants or Variant.CYTHON not in variants:
            logging.warning(
                f"Skipping benchmark '{benchmark_id}': Missing one variant."
            )
            continue

        py_item = variants[Variant.PYTHON]
        cy_item = variants[Variant.CYTHON]

        # Verify that both implementations use identical input parameters.
        if py_item.args != cy_item.args or py_item.kwargs != cy_item.kwargs:
            logging.warning(
                f"Skipping benchmark '{benchmark_id}': "
                "Mismatched input parameters between Python and Cython variants."
            )
            continue

        logging.info(f"Running benchmark: {benchmark_id}")
        py_avg, py_std = run_benchmark(py_item)
        cy_avg, cy_std = run_benchmark(cy_item)
        speedup = py_avg / cy_avg if cy_avg > 0 else float("inf")
        results.append(
            {
                "benchmark": benchmark_id,
                "py_avg": py_avg,
                "py_std": py_std,
                "cy_avg": cy_avg,
                "cy_std": cy_std,
                "speedup": speedup,
            }
        )
    return results


def generate_markdown_report(
    results: List[Dict[str, Any]], filename: str = "benchmarks.md"
) -> None:
    """Write out a Markdown report summarizing the benchmark results in a table.

    Args:
        results (List[Dict[str, Any]]): The list of benchmark results.
        filename (str): The filename for the generated Markdown report.
    """
    with open(filename, "w") as f:
        f.write("# Benchmark Report\n\n")
        f.write(
            "| Benchmark | Python Avg (s) | Python Std (s) | Cython Avg (s) | Cython Std (s) | Speedup |\n"
        )
        f.write(
            "|-----------|----------------|----------------|----------------|----------------|---------|\n"
        )
        for res in results:
            f.write(
                f"| {res['benchmark']} "
                f"| {res['py_avg']:.6f} "
                f"| {res['py_std']:.6f} "
                f"| {res['cy_avg']:.6f} "
                f"| {res['cy_std']:.6f} "
                f"| {res['speedup']:.2f}x |\n"
            )
    logging.info(f"Benchmark report saved to {filename}")


if __name__ == "__main__":
    # Import all submodules under 'cnake_charmer' so that benchmark decorators are executed.
    import_all_submodules("cnake_charmer")

    results = run_all_benchmarks()
    generate_markdown_report(results)
