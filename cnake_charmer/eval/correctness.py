"""
Check correctness of a compiled Cython module against a Python reference.

Runs both implementations with the same test inputs and compares outputs.
Execution happens in a sandboxed subprocess for safety.
"""

import importlib
import importlib.util
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CorrectnessResult:
    success: bool
    passed: int = 0
    total: int = 0
    score: float = 0.0  # 0.0 to 1.0
    failures: list = field(default_factory=list)
    error: str = ""


def _load_module_from_path(module_path: str, module_name: str = "loaded_module"):
    """Dynamically load a Python/Cython module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    # Add the module's directory to sys.path temporarily for deps
    module_dir = os.path.dirname(os.path.abspath(module_path))
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Runner script executed inside the sandbox
# ---------------------------------------------------------------------------

_RUNNER_SCRIPT = """\
import sys, json, os, importlib, importlib.util, math, types

def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    mod_dir = os.path.dirname(os.path.abspath(path))
    if mod_dir not in sys.path:
        sys.path.insert(0, mod_dir)
    spec.loader.exec_module(mod)
    return mod

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
        return all(_results_equal(x, y, rtol) for x, y in zip(a, b))
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
            return (tuple(test_case[0]) if isinstance(test_case[0], (list, tuple))
                    else (test_case[0],)), test_case[1]
        else:
            return tuple(test_case), {}
    else:
        return (test_case,), {}

# Read config from stdin
config = json.loads(sys.stdin.read())
python_code = config["python_code"]
func_name = config["func_name"]
cython_module_path = config["cython_module_path"]
test_cases = config["test_cases"]

# Load Python function
py_namespace = {}
exec(python_code, py_namespace)
python_func = py_namespace.get(func_name)
if python_func is None:
    print(json.dumps({"error": f"Function {func_name!r} not found in Python code"}))
    sys.exit(0)

# Load Cython module — name must match what's compiled into the .so
so_basename = os.path.basename(cython_module_path)
module_name = so_basename.split(".")[0]  # e.g. "gen_module" from "gen_module.cpython-312-x86_64-linux-gnu.so"
try:
    cy_module = _load_module(cython_module_path, module_name)
    cython_func = getattr(cy_module, func_name)
except Exception as e:
    print(json.dumps({"error": f"Failed to load Cython module: {e}"}))
    sys.exit(0)

# Run test cases
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
print(json.dumps({
    "passed": passed,
    "total": total,
    "score": score,
    "failures": failures,
}))
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_correctness(
    python_func=None,
    cython_module_path: str | None = None,
    cython_func=None,
    func_name: str | None = None,
    test_cases: list | None = None,
    timeout_per_case: float = 10.0,
    # New parameters for sandboxed execution
    python_code: str | None = None,
) -> CorrectnessResult:
    """
    Check that a Cython implementation produces the same outputs as Python.

    For sandboxed execution (preferred), provide:
    - python_code + func_name + cython_module_path + test_cases

    For legacy in-process execution (backward compat), provide:
    - python_func + cython_func (or cython_module_path + func_name)

    Args:
        python_func: The reference Python callable (legacy, in-process).
        python_code: Python source code string (sandboxed).
        cython_module_path: Path to compiled .so/.pyd file.
        cython_func: Direct callable (legacy, in-process).
        func_name: Function name to extract from the cython module.
        test_cases: List of test cases.
        timeout_per_case: Max seconds per test case.

    Returns:
        CorrectnessResult with pass/fail counts and details.
    """
    if test_cases is None or len(test_cases) == 0:
        return CorrectnessResult(success=False, error="No test cases provided")

    # Sandboxed path: when we have source code + module path
    if python_code and cython_module_path and func_name:
        return _check_correctness_sandboxed(
            python_code=python_code,
            func_name=func_name,
            cython_module_path=cython_module_path,
            test_cases=test_cases,
            timeout_per_case=timeout_per_case,
        )

    # Legacy in-process path (backward compat)
    if python_func is None:
        return CorrectnessResult(success=False, error="No Python reference function provided")

    return _check_correctness_inprocess(
        python_func=python_func,
        cython_module_path=cython_module_path,
        cython_func=cython_func,
        func_name=func_name,
        test_cases=test_cases,
    )


def _check_correctness_sandboxed(
    python_code: str,
    func_name: str,
    cython_module_path: str,
    test_cases: list,
    timeout_per_case: float = 10.0,
) -> CorrectnessResult:
    """Run correctness checks in a sandboxed subprocess."""
    from cnake_charmer.eval.sandbox import execute_config, run_python_sandboxed

    module_dir = os.path.dirname(os.path.abspath(cython_module_path))

    # Serialize test cases — handle numpy arrays by converting to lists
    serializable_cases = _make_serializable(test_cases)

    config_data = json.dumps(
        {
            "python_code": python_code,
            "func_name": func_name,
            "cython_module_path": cython_module_path,
            "test_cases": serializable_cases,
        }
    )

    # Build runner script that reads config from a file (not stdin for bwrap compat)
    config_script = f"""\
import sys, os
# Write config to a temp file for the runner to read
config_data = {config_data!r}
with open(os.path.join(os.path.dirname(__file__), '_config.json'), 'w') as f:
    f.write(config_data)
"""

    # Modify runner to read from file instead of stdin
    runner = _RUNNER_SCRIPT.replace(
        "config = json.loads(sys.stdin.read())",
        "config = json.loads(open(os.path.join(os.path.dirname(__file__), '_config.json')).read())",
    )

    full_script = config_script + "\n" + runner

    total_timeout = max(30, int(timeout_per_case * len(test_cases)) + 10)
    sandbox_cfg = execute_config(
        wall_time_limit_s=total_timeout,
        cpu_time_limit_s=total_timeout - 5,
        extra_ro_binds=(module_dir,),
    )

    result = run_python_sandboxed(
        full_script,
        config=sandbox_cfg,
    )

    if result.timed_out:
        return CorrectnessResult(
            success=False,
            error=f"Correctness check timed out after {total_timeout}s",
        )

    if result.returncode != 0:
        return CorrectnessResult(
            success=False,
            error=f"Correctness runner failed (rc={result.returncode}): {result.stderr[:500]}",
        )

    # Parse JSON output
    try:
        data = json.loads(result.stdout.strip())
    except (json.JSONDecodeError, ValueError):
        return CorrectnessResult(
            success=False,
            error=f"Failed to parse correctness output: {result.stdout[:500]}",
        )

    if "error" in data:
        return CorrectnessResult(success=False, error=data["error"])

    return CorrectnessResult(
        success=True,
        passed=data["passed"],
        total=data["total"],
        score=data["score"],
        failures=data.get("failures", []),
    )


def _check_correctness_inprocess(
    python_func,
    cython_module_path=None,
    cython_func=None,
    func_name=None,
    test_cases=None,
) -> CorrectnessResult:
    """Legacy in-process correctness checking (no sandboxing)."""
    if cython_func is None:
        if cython_module_path is None:
            return CorrectnessResult(
                success=False, error="No Cython function or module path provided"
            )
        if func_name is None:
            return CorrectnessResult(
                success=False, error="func_name required when loading from module path"
            )
        try:
            module = _load_module_from_path(cython_module_path, "cython_test_module")
            cython_func = getattr(module, func_name)
        except Exception as e:
            return CorrectnessResult(success=False, error=f"Failed to load Cython module: {e}")

    result = CorrectnessResult(success=True, total=len(test_cases))

    for i, test_case in enumerate(test_cases):
        args, kwargs = _unpack_test_case(test_case)

        try:
            py_result = python_func(*args, **kwargs)
        except Exception as e:
            result.failures.append(f"Case {i}: Python reference raised {type(e).__name__}: {e}")
            continue

        try:
            cy_result = cython_func(*args, **kwargs)
        except Exception as e:
            result.failures.append(f"Case {i}: Cython raised {type(e).__name__}: {e}")
            continue

        if not _results_equal(py_result, cy_result):
            result.failures.append(
                f"Case {i}: Output mismatch. Python={_truncate(py_result)}, Cython={_truncate(cy_result)}"
            )
        else:
            result.passed += 1

    result.score = result.passed / result.total if result.total > 0 else 0.0
    if result.failures:
        result.success = True  # Partial success is still "ran successfully"
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_serializable(test_cases: list) -> list:
    """Convert test cases to JSON-serializable format (numpy arrays → lists)."""
    try:
        import numpy as np

        has_numpy = True
    except ImportError:
        has_numpy = False

    def _convert(obj):
        if has_numpy and isinstance(obj, np.ndarray):
            return {"__ndarray__": True, "data": obj.tolist(), "dtype": str(obj.dtype)}
        if isinstance(obj, (list, tuple)):
            return [_convert(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        return obj

    return _convert(test_cases)


def _unpack_test_case(test_case):
    """Unpack a test case into (args, kwargs)."""
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
            return tuple(test_case[0]) if isinstance(test_case[0], (list, tuple)) else (
                test_case[0],
            ), test_case[1]
        else:
            # Treat the whole tuple as args
            return tuple(test_case), {}
    else:
        return (test_case,), {}


def _results_equal(a: Any, b: Any, rtol: float = 1e-6) -> bool:
    """Compare two results, handling floats, lists, and numpy arrays."""
    if type(a) is not type(b):
        # Allow int/float comparison
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
        return all(_results_equal(x, y, rtol) for x, y in zip(a, b, strict=True))

    # Try numpy array comparison
    try:
        import numpy as np

        if isinstance(a, np.ndarray):
            return np.allclose(a, b, rtol=rtol)
    except ImportError:
        pass

    return a == b


def _truncate(value: Any, max_len: int = 100) -> str:
    """Truncate a value's repr for error messages."""
    s = repr(value)
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s
