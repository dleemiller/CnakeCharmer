"""Sandbox runner: correctness checking.

Compares a Python reference function against a compiled Cython module
on a list of test cases and reports pass/fail counts.

SANDBOX CONSTRAINTS — read before editing:

    1. This script runs inside a bwrap sandbox.  It MUST NOT import
       from cnake_charmer — the repo root is not mounted.
    2. Only stdlib, numpy, and venv packages are available.
    3. Shared helpers live in _common.py (same directory, auto on sys.path).
    4. Config JSON path is sys.argv[1].
    5. Output is a single JSON object on stdout.

Expected config keys:
    python_code          str   — Python source code containing the reference function
    func_name            str   — name of the function to compare
    cython_module_path   str   — absolute path to the compiled .so
    test_cases           list  — each element is args-list, (args, kwargs) pair, or dict

Output JSON:
    {"passed": int, "total": int, "score": float, "failures": [str, ...]}
    or on load error:
    {"error": str}
"""

import json

from _common import load_both_funcs, load_config

# ── helpers (self-contained, no external deps) ────────────────────────


def _results_equal(a, b, rtol=1e-6):
    if type(a) is not type(b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if isinstance(a, float) or isinstance(b, float):
                return abs(float(a) - float(b)) <= rtol * max(abs(float(a)), abs(float(b)), 1.0)
            return a == b
        return False
    if isinstance(a, float):
        return abs(a - b) <= rtol * max(abs(a), abs(b), 1.0)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_results_equal(x, y, rtol) for x, y in zip(a, b, strict=False))
    try:
        import numpy as np

        if isinstance(a, np.ndarray):
            return bool(np.allclose(a, b, rtol=rtol))
    except ImportError:
        pass
    return a == b


def _truncate(value, max_len=100):
    s = repr(value)
    return s[:max_len] + "..." if len(s) > max_len else s


def _unpack_test_case(test_case):
    if isinstance(test_case, dict):
        return test_case.get("args", ()), test_case.get("kwargs", {})
    elif isinstance(test_case, (list, tuple)):
        if len(test_case) == 0:
            return (), {}
        elif len(test_case) == 1:
            args = test_case[0]
            if isinstance(args, (list, tuple)):
                return tuple(args), {}
            return (args,), {}
        elif len(test_case) == 2 and isinstance(test_case[1], dict):
            return (
                tuple(test_case[0]) if isinstance(test_case[0], (list, tuple)) else (test_case[0],),
                test_case[1],
            )
        else:
            return tuple(test_case), {}
    else:
        return (test_case,), {}


# ── main ──────────────────────────────────────────────────────────────

config = load_config()
python_func, cython_func = load_both_funcs(config)
test_cases = config["test_cases"]

passed = 0
total = len(test_cases)
failures = []

for i, test_case in enumerate(test_cases):
    args, kwargs = _unpack_test_case(test_case)
    try:
        py_result = python_func(*args, **kwargs)
    except Exception as e:
        failures.append(f"Case {i}: Python reference raised {type(e).__name__}: {e}")
        continue
    try:
        cy_result = cython_func(*args, **kwargs)
    except Exception as e:
        failures.append(f"Case {i}: Cython raised {type(e).__name__}: {e}")
        continue
    if not _results_equal(py_result, cy_result):
        failures.append(
            f"Case {i}: Output mismatch. Python={_truncate(py_result)}, "
            f"Cython={_truncate(cy_result)}"
        )
    else:
        passed += 1

score = passed / total if total > 0 else 0.0
print(json.dumps({"passed": passed, "total": total, "score": score, "failures": failures}))
