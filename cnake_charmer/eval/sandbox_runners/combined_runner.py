"""Sandbox runner: combined test + benchmark for the training environment.

Runs assertion-style equivalence tests (``py.func() == cy.func()``) and
optionally benchmarks the first matching function.  Used by
``CythonToolEnvironment.evaluate_cython()`` during SFT/GRPO training.

SANDBOX CONSTRAINTS — read before editing:

    1. This script runs inside a bwrap sandbox.  It MUST NOT import
       from cnake_charmer — the repo root is not mounted.
    2. Only stdlib, numpy, and venv packages are available.
    3. Shared helpers live in _common.py (same directory, auto on sys.path).
    4. Config JSON path is sys.argv[1].
    5. Output is a single JSON object on stdout.

Expected config keys:
    python_code          str  — Python source code (reference implementation)
    cython_module_path   str  — absolute path to compiled .so
    test_code            str  — multi-line assertion code (py.X(...) == cy.X(...))
    do_benchmark         bool — whether to run benchmarks after tests (default true)

Output JSON:
    {"py_loaded": bool, "cy_loaded": bool,
     "tests": {"passed": int, "total": int, "failures": [str, ...]},
     "benchmark": {"success": bool, "speedup": float, ...} | null}
    or on load error:
    {"py_loaded": bool, "cy_loaded": bool, "py_error"|"cy_error": str}
"""

import contextlib
import json
import math
import os
import re
import signal
import sys
import time
import types

from _common import load_config, load_module_from_path

# ── timeout context manager ───────────────────────────────────────────

ASSERTION_TIMEOUT = 5


class _TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TimeoutError("Timed out")


@contextlib.contextmanager
def _alarm(seconds):
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ── module loading ────────────────────────────────────────────────────


def _exec_as_module(code, name="dynamic_module"):
    mod = types.ModuleType(name)
    exec(code, mod.__dict__)  # noqa: S102
    return mod


def _approx_equal(left, right, rel_tol=1e-9, abs_tol=1e-12):
    """Compare values with float tolerance, including nested containers."""
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right

    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=rel_tol, abs_tol=abs_tol)

    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(
            _approx_equal(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
            for a, b in zip(left, right, strict=False)
        )

    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return False
        return all(_approx_equal(left[k], right[k], rel_tol=rel_tol, abs_tol=abs_tol) for k in left)

    if hasattr(left, "shape") and hasattr(right, "shape"):
        try:
            import numpy as np

            return bool(np.allclose(left, right, rtol=rel_tol, atol=abs_tol, equal_nan=True))
        except Exception:
            return False

    return left == right


# ── test runner ───────────────────────────────────────────────────────


def run_tests(py_mod, cy_mod, test_code):
    namespace = {"py": py_mod, "cy": cy_mod}
    passed = 0
    total = 0
    failures = []

    for line in test_code.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        is_assert = "==" in line
        try:
            with _alarm(ASSERTION_TIMEOUT):
                if is_assert:
                    total += 1
                    parts = line.split("==", 1)
                    left = eval(parts[0].strip(), namespace)  # noqa: S307
                    right = eval(parts[1].strip(), namespace)  # noqa: S307
                    if _approx_equal(left, right):
                        passed += 1
                    else:
                        failures.append(
                            f"FAIL: {line}\n"
                            f"  left:  {repr(left)[:200]}\n"
                            f"  right: {repr(right)[:200]}"
                        )
                else:
                    exec(line, namespace)  # noqa: S102
        except _TimeoutError:
            if is_assert:
                failures.append(f"TIMEOUT: {line} (>{ASSERTION_TIMEOUT}s)")
            else:
                failures.append(f"TIMEOUT (setup): {line}")
        except Exception as e:
            if is_assert:
                failures.append(f"ERROR: {line}\n  {type(e).__name__}: {e}")
            else:
                failures.append(f"ERROR (setup): {line}\n  {type(e).__name__}: {e}")

    return {"passed": passed, "total": total, "failures": failures}


# ── benchmark ─────────────────────────────────────────────────────────


def run_benchmark(py_mod, cy_mod, test_code):
    for line in test_code.strip().splitlines():
        line = line.strip()
        m = re.match(r"py\.(\w+)\((.+?)\)\s*==\s*cy\.\w+\(", line)
        if m:
            func_name = m.group(1)
            py_func = getattr(py_mod, func_name, None)
            cy_func = getattr(cy_mod, func_name, None)
            if py_func and cy_func:
                args_str = m.group(2)
                try:
                    args = eval(f"({args_str},)")  # noqa: S307
                except Exception:
                    continue
                try:
                    return _time_benchmark(py_func, cy_func, args, num_runs=3)
                except Exception as e:
                    return {"success": False, "error": str(e)}
    return None


def _time_benchmark(py_func, cy_func, args, num_runs=3, max_total=5.0):
    # Warmup
    for _ in range(2):
        py_func(*args)
        cy_func(*args)

    def _time(func):
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            func(*args)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            if sum(times) > max_total and len(times) >= 2:
                break
        return sum(times) / len(times)

    py_time = _time(py_func)
    cy_time = _time(cy_func)
    speedup = py_time / cy_time if cy_time > 0 else float("inf")

    return {
        "success": True,
        "speedup": round(speedup, 2),
        "cython_time": round(cy_time, 6),
        "python_time": round(py_time, 6),
        "errors": "",
    }


# ── main ──────────────────────────────────────────────────────────────

config = load_config()

python_code = config["python_code"]
cython_module_path = config["cython_module_path"]
test_code = config["test_code"]
do_benchmark = config.get("do_benchmark", True)

output = {"py_loaded": False, "cy_loaded": False}

# Load modules
try:
    py_mod = _exec_as_module(python_code, "py_module")
    output["py_loaded"] = True
except Exception as e:
    output["py_error"] = f"Failed to load Python code: {e}"
    print(json.dumps(output))
    sys.exit(0)

so_basename = os.path.basename(cython_module_path)
module_name = so_basename.split(".")[0]
try:
    cy_mod = load_module_from_path(cython_module_path, module_name)
    output["cy_loaded"] = True
except Exception as e:
    output["cy_error"] = f"Failed to load compiled Cython: {e}"
    print(json.dumps(output))
    sys.exit(0)

# Run tests
test_results = run_tests(py_mod, cy_mod, test_code)
output["tests"] = test_results

# Run benchmark (only if tests passed)
if do_benchmark and test_results["total"] > 0 and test_results["passed"] > 0:
    bench_result = run_benchmark(py_mod, cy_mod, test_code)
    if bench_result:
        output["benchmark"] = bench_result

print(json.dumps(output))
