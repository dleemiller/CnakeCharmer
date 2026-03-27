#!/usr/bin/env python
"""
Benchmark Runner for Python vs. Cython Implementations.

Uses source file hashing to skip unchanged benchmarks.
Saves cache incrementally. Runs up to 4 benchmarks in parallel.

Usage:
    uv run --no-sync run_benchmarks.py          # only changed, 4 workers
    uv run --no-sync run_benchmarks.py --all    # force re-run everything
    uv run --no-sync run_benchmarks.py -j 1     # single-threaded
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
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _run_single_benchmark(benchmark_id: str) -> dict:
    """Run a single benchmark in a subprocess. Returns results dict."""
    # Re-import everything in this process
    import_all_submodules("cnake_charmer")
    from cnake_charmer.benchmarks.registry import Variant
    from cnake_charmer.benchmarks.registry import benchmark_registry as reg

    variants = reg.get(benchmark_id, {})
    python_variant = variants.get(Variant.PYTHON)
    cython_variant = variants.get(Variant.CYTHON)
    purepy_variant = variants.get(Variant.CYTHON_PP)
    simd_variant = variants.get(Variant.CYTHON_SIMD)

    if python_variant is None:
        return {"benchmark_id": benchmark_id, "results": [], "error": "no python variant"}

    py_results = None
    new_results = []
    cat = python_variant.category if python_variant.category else ""

    for label, variant_item in [
        ("cython", cython_variant),
        ("pure py", purepy_variant),
        ("simd", simd_variant),
    ]:
        if variant_item is None:
            continue
        if python_variant.args != variant_item.args or python_variant.kwargs != variant_item.kwargs:
            continue

        if py_results is None:
            py_results = run_benchmark(python_variant)

        variant_avg, variant_std = run_benchmark(variant_item)
        speedup = py_results[0] / variant_avg if variant_avg > 0 else float("inf")
        new_results.append(
            {
                "benchmark": benchmark_id,
                "category": cat,
                "syntax": label,
                "py_avg": py_results[0],
                "py_std": py_results[1],
                "cy_avg": variant_avg,
                "cy_std": variant_std,
                "speedup": speedup,
            }
        )

    return {"benchmark_id": benchmark_id, "results": new_results}


def run_all_benchmarks(force_all: bool = False, num_workers: int = 4) -> list[dict[str, Any]]:
    cache = load_cache()
    results: list[dict[str, Any]] = []
    to_run: list[str] = []
    skipped = 0

    for benchmark_id, variants in benchmark_registry.items():
        python_variant = variants.get(Variant.PYTHON)
        cython_variant = variants.get(Variant.CYTHON)
        purepy_variant = variants.get(Variant.CYTHON_PP)
        simd_variant = variants.get(Variant.CYTHON_SIMD)

        if python_variant is None or (
            cython_variant is None and purepy_variant is None and simd_variant is None
        ):
            continue

        current_hash = _compute_hash(benchmark_id)
        cached_hash = cache["hashes"].get(benchmark_id, "")

        cached_results = cache["results"].get(benchmark_id, [])
        cached_labels = {r["syntax"] for r in cached_results}
        expected_labels = {
            label
            for label, v in [
                ("cython", cython_variant),
                ("pure py", purepy_variant),
                ("simd", simd_variant),
            ]
            if v is not None
        }

        if not force_all and current_hash == cached_hash and expected_labels <= cached_labels:
            for cached_result in cached_results:
                results.append(cached_result)
            skipped += 1
            continue

        to_run.append(benchmark_id)

    if not to_run:
        log.info(f"Skipped all {skipped} benchmarks (use --all to force)")
        return results

    log.info(f"Running {len(to_run)} benchmarks with {num_workers} workers, {skipped} cached")
    for bid in to_run:
        log.info(f"  → {bid}")

    if num_workers <= 1:
        # Single-threaded fallback
        for idx, bid in enumerate(to_run, 1):
            log.info(f"[{idx}/{len(to_run)}] [dim]starting[/dim] {bid}...", extra={"markup": True})
            result = _run_single_benchmark(bid)
            for entry in result["results"]:
                results.append(entry)
                log.info(
                    f"[{idx}/{len(to_run)}] {bid} ({entry['syntax']}): "
                    f"{entry['py_avg'] * 1000:.1f}ms → {entry['cy_avg'] * 1000:.1f}ms = "
                    f"[bold]{entry['speedup']:.1f}x[/bold]",
                    extra={"markup": True},
                )
            cache["hashes"][bid] = _compute_hash(bid)
            cache["results"][bid] = result["results"]
            save_cache(cache)
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_run_single_benchmark, bid): bid for bid in to_run}
            for done_count, future in enumerate(as_completed(futures), 1):
                bid = futures[future]
                try:
                    result = future.result()
                    for entry in result["results"]:
                        results.append(entry)
                        log.info(
                            f"[{done_count}/{len(to_run)}] {bid} ({entry['syntax']}): "
                            f"{entry['py_avg'] * 1000:.1f}ms → {entry['cy_avg'] * 1000:.1f}ms = "
                            f"[bold]{entry['speedup']:.1f}x[/bold]",
                            extra={"markup": True},
                        )
                    cache["hashes"][bid] = _compute_hash(bid)
                    cache["results"][bid] = result["results"]
                    save_cache(cache)
                except Exception as e:
                    log.error(f"[{done_count}/{len(to_run)}] {bid} FAILED: {e}")

    log.info(f"Ran {len(to_run)} benchmarks, {skipped} cached")
    return results


def generate_markdown_report(
    results: list[dict[str, Any]], filename: str = "benchmarks.md"
) -> None:
    with open(filename, "w") as f:
        f.write("# Benchmark Report\n\n")
        f.write("| Category | Benchmark | Variant | Python (ms) | Cython (ms) | Speedup |\n")
        f.write("|----------|-----------|---------|-------------|-------------|----------|\n")
        for res in sorted(results, key=lambda r: r["speedup"], reverse=True):
            f.write(
                f"| {res.get('category', '')} "
                f"| {res['benchmark']} "
                f"| {res['syntax']} "
                f"| {res['py_avg'] * 1000:.3f} "
                f"| {res['cy_avg'] * 1000:.3f} "
                f"| {res['speedup']:.1f}x |\n"
            )

    table = Table(title=f"Benchmarks ({len(results)} results)")
    table.add_column("Category", style="dim")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Variant", style="dim")
    table.add_column("Python (ms)", justify="right")
    table.add_column("Cython (ms)", justify="right")
    table.add_column("Speedup", justify="right", style="bold green")

    for res in sorted(results, key=lambda r: r["speedup"], reverse=True)[:20]:
        table.add_row(
            res.get("category", ""),
            res["benchmark"],
            res["syntax"],
            f"{res['py_avg'] * 1000:.1f}",
            f"{res['cy_avg'] * 1000:.1f}",
            f"{res['speedup']:.1f}x",
        )

    if len(results) > 20:
        table.add_row("", "...", "", "", "", f"({len(results) - 20} more)")

    console.print(table)
    log.info(f"Report saved to {filename}")


def append_kernel_report(filename: str = "benchmarks.md") -> None:
    """Append kernel-only benchmark table to the report.

    Uses engine kernels (pre-allocated tensors, no allocation in timing).
    Compares portable Cython (scalar) vs platform SIMD (AVX2+FMA or NEON).
    Silently skips if engine kernels are not built.
    """
    try:
        from cnake_charmer.engine.kernels.bench_wrapper import bench_all_kernels
    except ImportError:
        log.info("Engine kernels not built, skipping kernel-only table")
        return

    log.info("Running kernel-only benchmarks (inference mode)...")
    kernel_results = bench_all_kernels()

    if not kernel_results:
        return

    with open(filename, "a") as f:
        f.write("\n\n## Kernel-Only Benchmark (Inference Mode)\n\n")
        f.write("Pre-allocated tensors, timing only the compute kernel.\n")
        f.write("Compares portable Cython (scalar) vs platform-optimized SIMD.\n\n")
        f.write("| Kernel | Size | Portable (ms) | SIMD (ms) | SIMD ISA | Speedup |\n")
        f.write("|--------|------|--------------|-----------|----------|----------|\n")
        for r in kernel_results:
            if r["simd_ms"] is not None:
                speedup = r["scalar_ms"] / r["simd_ms"] if r["simd_ms"] > 0 else 0
                f.write(
                    f"| {r['kernel']} "
                    f"| {r['size']} "
                    f"| {r['scalar_ms']:.3f} "
                    f"| {r['simd_ms']:.3f} "
                    f"| {r['simd_label']} "
                    f"| {speedup:.1f}x |\n"
                )
            else:
                f.write(
                    f"| {r['kernel']} "
                    f"| {r['size']} "
                    f"| {r['scalar_ms']:.3f} "
                    f"| — "
                    f"| {r['simd_label']} "
                    f"| — |\n"
                )

    # Print to console
    ktable = Table(title="Kernel-Only (Inference Mode)")
    ktable.add_column("Kernel", style="cyan")
    ktable.add_column("Size")
    ktable.add_column("Portable (ms)", justify="right")
    ktable.add_column("SIMD (ms)", justify="right")
    ktable.add_column("ISA", style="dim")
    ktable.add_column("Speedup", justify="right", style="bold green")

    for r in kernel_results:
        simd_str = f"{r['simd_ms']:.3f}" if r["simd_ms"] is not None else "—"
        speedup_str = (
            f"{r['scalar_ms'] / r['simd_ms']:.1f}x" if r["simd_ms"] and r["simd_ms"] > 0 else "—"
        )
        ktable.add_row(
            r["kernel"],
            r["size"],
            f"{r['scalar_ms']:.3f}",
            simd_str,
            r["simd_label"],
            speedup_str,
        )

    console.print(ktable)
    log.info("Kernel-only results appended to benchmarks.md")


if __name__ == "__main__":
    force = "--all" in sys.argv

    # Parse -j N for worker count
    num_workers = 4
    for i, arg in enumerate(sys.argv):
        if arg == "-j" and i + 1 < len(sys.argv):
            num_workers = int(sys.argv[i + 1])

    import_all_submodules("cnake_charmer")

    results = run_all_benchmarks(force_all=force, num_workers=num_workers)
    generate_markdown_report(results)
    append_kernel_report()
