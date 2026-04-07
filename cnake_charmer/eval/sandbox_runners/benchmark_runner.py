"""Sandbox runner: benchmarking.

Times a Python reference function and a compiled Cython module and
reports the speedup ratio.

SANDBOX CONSTRAINTS — read before editing:

    1. This script runs inside a bwrap sandbox.  It MUST NOT import
       from cnake_charmer — the repo root is not mounted.
    2. Only stdlib, numpy, and venv packages are available.
    3. Shared helpers live in _common.py (same directory, auto on sys.path).
    4. Config JSON path is sys.argv[1].
    5. Output is a single JSON object on stdout.

Expected config keys:
    python_code          str   — Python source code containing the reference function
    func_name            str   — name of the function to benchmark
    cython_module_path   str   — absolute path to the compiled .so
    args                 list  — positional arguments (default [])
    kwargs               dict  — keyword arguments (default {})
    num_runs             int   — timed iterations (default 10)
    warmup_runs          int   — untimed warmup iterations (default 2)
    max_total_seconds    float — per-function time cap (default 5.0)

Output JSON:
    {"success": true, "speedup": float, "python_time": float,
     "cython_time": float, "python_std": float, "cython_std": float,
     "num_runs": int}
    or on error:
    {"error": str}
"""

import json
import statistics
import sys
import time

from _common import apply_rlimits, load_cython_func, load_python_func

# ── helpers ───────────────────────────────────────────────────────────


def _time_function(func, args, kwargs, num_runs, max_total):
    times = []
    cumulative = 0.0
    for _ in range(num_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        cumulative += elapsed
        if cumulative > max_total and len(times) >= 2:
            break
    return times


# ── main ──────────────────────────────────────────────────────────────

config = json.loads(open(sys.argv[1]).read())  # noqa: SIM115
apply_rlimits(config)

python_code = config["python_code"]
func_name = config["func_name"]
cython_module_path = config["cython_module_path"]
args = tuple(config.get("args", ()))
kwargs = config.get("kwargs", {})
num_runs = config.get("num_runs", 10)
warmup_runs = config.get("warmup_runs", 2)
max_total_seconds = config.get("max_total_seconds", 5.0)

# Load Python function
try:
    python_func = load_python_func(python_code, func_name)
except Exception as e:
    print(json.dumps({"error": f"Function {func_name!r} not found in Python code: {e}"}))
    sys.exit(0)

# Load Cython module
try:
    cython_func = load_cython_func(cython_module_path, func_name)
except Exception as e:
    print(json.dumps({"error": f"Failed to load Cython module: {e}"}))
    sys.exit(0)

# Warmup
try:
    for _ in range(warmup_runs):
        start = time.perf_counter()
        python_func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        if elapsed > max_total_seconds:
            num_runs = min(num_runs, 2)
        cython_func(*args, **kwargs)
except Exception as e:
    print(json.dumps({"error": f"Warmup failed: {e}"}))
    sys.exit(0)

# Time Python
try:
    py_times = _time_function(python_func, args, kwargs, num_runs, max_total_seconds)
except Exception as e:
    print(json.dumps({"error": f"Python benchmark failed: {e}"}))
    sys.exit(0)

# Time Cython
try:
    cy_times = _time_function(cython_func, args, kwargs, num_runs, max_total_seconds)
except Exception as e:
    print(json.dumps({"error": f"Cython benchmark failed: {e}"}))
    sys.exit(0)

py_mean = statistics.mean(py_times)
cy_mean = statistics.mean(cy_times)
py_std = statistics.stdev(py_times) if len(py_times) > 1 else 0.0
cy_std = statistics.stdev(cy_times) if len(cy_times) > 1 else 0.0
speedup = py_mean / cy_mean if cy_mean > 0 else float("inf")

print(
    json.dumps(
        {
            "success": True,
            "speedup": speedup,
            "python_time": py_mean,
            "cython_time": cy_mean,
            "python_std": py_std,
            "cython_std": cy_std,
            "num_runs": len(py_times),
        }
    )
)
