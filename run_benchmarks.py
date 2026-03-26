#!/usr/bin/env python
"""
Benchmark Runner for Python vs. Cython Implementations.

Uses source file hashing to skip unchanged benchmarks.
Saves cache incrementally after each benchmark so kills don't lose progress.

Usage:
    uv run python run_benchmarks.py          # only run changed benchmarks
    uv run python run_benchmarks.py --all    # force re-run everything
"""

import contextlib
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

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from cnake_charmer.benchmarks.registry import BenchmarkItem, Variant, benchmark_registry

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, show_time=False, show_path=False)],
)
log = logging.getLogger("benchmark")

CACHE_FILE = Path(".benchmark_cache.json")
PY_DIR = Path("cnake_charmer/py")
CY_DIR = Path("cnake_charmer/cy")
PP_DIR = Path("cnake_charmer/pp")
SIMD_DIR = Path("cnake_charmer/cy_simd")


def import_all_submodules(package_name: str) -> None:
    """Dynamically import all submodules so benchmark decorators register."""
    package = importlib.import_module(package_name)
    for _loader, module_name, _is_pkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        with contextlib.suppress(Exception):
            importlib.import_module(module_name)


def _hash_file(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _find_source_files(benchmark_id: str) -> list[Path]:
    files = []
    for base_dir in [PY_DIR, CY_DIR, PP_DIR, SIMD_DIR]:
        for ext in ["*.py", "*.pyx"]:
            for f in base_dir.rglob(ext):
                if f.stem == benchmark_id and f.name != "__init__.py":
                    files.append(f)
    return files


def _compute_hash(benchmark_id: str) -> str:
    files = sorted(_find_source_files(benchmark_id))
    if not files:
        return ""
    combined = "|".join(f"{f}:{_hash_file(f)}" for f in files)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"hashes": {}, "results": {}}


def save_cache(cache: dict) -> None:
    """Save benchmark cache atomically."""
    tmp = CACHE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=2))
    tmp.rename(CACHE_FILE)


def run_benchmark(item: BenchmarkItem) -> tuple[float, float]:
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
    cache = load_cache()
    results: list[dict[str, Any]] = []
    skipped = 0
    ran = 0
    total = len(benchmark_registry)

    for idx, (benchmark_id, variants) in enumerate(benchmark_registry.items(), 1):
        python_variant = variants.get(Variant.PYTHON)
        cython_variant = variants.get(Variant.CYTHON)
        purepy_variant = variants.get(Variant.CYTHON_PP)
        simd_variant = variants.get(Variant.CYTHON_SIMD)

        if python_variant is None or (
            cython_variant is None and purepy_variant is None and simd_variant is None
        ):
            continue

        # Check if sources changed
        current_hash = _compute_hash(benchmark_id)
        cached_hash = cache["hashes"].get(benchmark_id, "")

        if not force_all and current_hash == cached_hash and benchmark_id in cache["results"]:
            for cached_result in cache["results"][benchmark_id]:
                results.append(cached_result)
            skipped += 1
            continue

        # Run the benchmark
        py_results = None
        new_results = []

        for label, variant_item in [
            ("cython", cython_variant),
            ("pure py", purepy_variant),
            ("simd", simd_variant),
        ]:
            if variant_item is None:
                continue
            if (
                python_variant.args != variant_item.args
                or python_variant.kwargs != variant_item.kwargs
            ):
                continue

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

            ran += 1
            log.info(
                f"[{idx}/{total}] {benchmark_id} ({label}): "
                f"{py_results[0] * 1000:.1f}ms → {variant_avg * 1000:.1f}ms = "
                f"[bold]{speedup:.1f}x[/bold]",
                extra={"markup": True},
            )

        # Save cache incrementally after each benchmark
        cache["hashes"][benchmark_id] = current_hash
        cache["results"][benchmark_id] = new_results
        save_cache(cache)

    if skipped:
        log.info(f"Skipped {skipped} unchanged benchmarks (use --all to force)")
    log.info(f"Ran {ran} benchmarks, {skipped} cached")

    return results


def generate_markdown_report(
    results: list[dict[str, Any]], filename: str = "benchmarks.md"
) -> None:
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

    # Also print a rich table summary
    table = Table(title=f"Benchmarks ({len(results)} results)")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Variant", style="dim")
    table.add_column("Python (ms)", justify="right")
    table.add_column("Cython (ms)", justify="right")
    table.add_column("Speedup", justify="right", style="bold green")

    for res in sorted(results, key=lambda r: r["speedup"], reverse=True)[:20]:
        table.add_row(
            res["benchmark"],
            res["syntax"],
            f"{res['py_avg'] * 1000:.1f}",
            f"{res['cy_avg'] * 1000:.1f}",
            f"{res['speedup']:.1f}x",
        )

    if len(results) > 20:
        table.add_row("...", "", "", "", f"({len(results) - 20} more)")

    console.print(table)
    log.info(f"Report saved to {filename}")


if __name__ == "__main__":
    force = "--all" in sys.argv

    import_all_submodules("cnake_charmer")

    results = run_all_benchmarks(force_all=force)
    generate_markdown_report(results)
