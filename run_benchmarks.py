#!/usr/bin/env python
"""
Benchmark Runner for Python vs. Cython Implementations.

Dynamically imports all submodules under 'cnake_charmer' to populate the
benchmark registry, then runs each pair. Uses source file hashing to skip
benchmarks that haven't changed since the last run.

Usage:
    uv run python run_benchmarks.py          # only run changed benchmarks
    uv run python run_benchmarks.py --all    # force re-run everything
"""

import hashlib
import importlib
import json
import logging
import pkgutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from cnake_charmer.benchmarks.registry import BenchmarkItem, Variant, benchmark_registry

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

CACHE_FILE = Path(".benchmark_cache.json")
PY_DIR = Path("cnake_charmer/py")
CY_DIR = Path("cnake_charmer/cy")
PP_DIR = Path("cnake_charmer/pp")


def import_all_submodules(package_name: str) -> None:
    """Dynamically import all submodules so benchmark decorators register."""
    package = importlib.import_module(package_name)
    for _loader, module_name, _is_pkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        try:
            importlib.import_module(module_name)
        except Exception as e:
            logging.warning(f"Could not import module {module_name}: {e}")


def _hash_file(path: Path) -> str:
    """SHA256 hash of a file's contents."""
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _find_source_files(benchmark_id: str) -> list[Path]:
    """Find the .py and .pyx source files for a benchmark by name."""
    files = []
    for base_dir in [PY_DIR, CY_DIR, PP_DIR]:
        for ext in ["*.py", "*.pyx"]:
            for f in base_dir.rglob(ext):
                if f.stem == benchmark_id and f.name != "__init__.py":
                    files.append(f)
    return files


def _compute_hash(benchmark_id: str) -> str:
    """Compute a combined hash of all source files for a benchmark."""
    files = sorted(_find_source_files(benchmark_id))
    if not files:
        return ""
    combined = "|".join(f"{f}:{_hash_file(f)}" for f in files)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def load_cache() -> dict:
    """Load cached benchmark results and hashes."""
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"hashes": {}, "results": {}}


def save_cache(cache: dict) -> None:
    """Save benchmark cache."""
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


def run_benchmark(item: BenchmarkItem) -> tuple[float, float]:
    """Run a benchmark and return (avg_time, std_dev)."""
    func = item.func
    args = item.args
    kwargs = item.kwargs
    num_runs = item.num_runs

    func(*args, **kwargs)  # warmup
    times: list[float] = []
    for _ in range(num_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        times.append(time.perf_counter() - start)
    avg = sum(times) / num_runs
    std = statistics.stdev(times) if num_runs > 1 else 0.0
    return avg, std


def run_all_benchmarks(force_all: bool = False) -> list[dict[str, Any]]:
    """Run benchmarks, skipping unchanged ones unless force_all is set."""
    cache = load_cache()
    results: list[dict[str, Any]] = []
    skipped = 0

    for benchmark_id, variants in benchmark_registry.items():
        python_variant = variants.get(Variant.PYTHON)
        cython_variant = variants.get(Variant.CYTHON)
        purepy_variant = variants.get(Variant.CYTHON_PP)

        if python_variant is None or (cython_variant is None and purepy_variant is None):
            logging.warning(f"Skipping benchmark '{benchmark_id}': Missing one variant.")
            continue

        # Check if sources changed
        current_hash = _compute_hash(benchmark_id)
        cached_hash = cache["hashes"].get(benchmark_id, "")

        if not force_all and current_hash == cached_hash and benchmark_id in cache["results"]:
            # Use cached results
            for cached_result in cache["results"][benchmark_id]:
                results.append(cached_result)
            skipped += 1
            continue

        # Run the benchmark
        py_results = None
        new_results = []

        for label, variant_item in [("cython", cython_variant), ("pure py", purepy_variant)]:
            if variant_item is None:
                continue
            if (
                python_variant.args != variant_item.args
                or python_variant.kwargs != variant_item.kwargs
            ):
                logging.warning(
                    f"Skipping benchmark '{benchmark_id}' ({label}): Mismatched input parameters."
                )
                continue

            logging.info(f"Running benchmark: {benchmark_id} ({label})")
            if py_results is None:
                py_results = run_benchmark(python_variant)
            variant_avg, variant_std = run_benchmark(variant_item)
            speedup = py_results[0] / variant_avg if variant_avg > 0 else float("inf")
            entry = {
                "benchmark": benchmark_id,
                "syntax": label,
                "py_avg": py_results[0],
                "py_std": py_results[1],
                "cy_avg": variant_avg,
                "cy_std": variant_std,
                "speedup": speedup,
            }
            new_results.append(entry)
            results.append(entry)

        # Update cache
        cache["hashes"][benchmark_id] = current_hash
        cache["results"][benchmark_id] = new_results

    if skipped:
        logging.info(f"Skipped {skipped} unchanged benchmarks (use --all to force)")

    save_cache(cache)
    return results


def generate_markdown_report(
    results: list[dict[str, Any]], filename: str = "benchmarks.md"
) -> None:
    """Write a Markdown report sorted by speedup."""
    with open(filename, "w") as f:
        f.write("# Benchmark Report\n\n")
        f.write("| Benchmark | Variant | Python (ms) | Cython (ms) | Speedup |\n")
        f.write("|-----------|---------|-------------|-------------|----------|\n")
        for res in sorted(results, key=lambda r: r["speedup"], reverse=True):
            f.write(
                f"| {res['benchmark']} "
                f"| {res['syntax']} "
                f"| {res['py_avg'] * 1000:.3f} "
                f"| {res['cy_avg'] * 1000:.3f} "
                f"| {res['speedup']:.1f}x |\n"
            )
    logging.info(f"Benchmark report saved to {filename}")


if __name__ == "__main__":
    force = "--all" in sys.argv

    import_all_submodules("cnake_charmer")

    results = run_all_benchmarks(force_all=force)
    generate_markdown_report(results)
