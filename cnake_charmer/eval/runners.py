"""
Sandboxed runner scripts for the training environment.

These scripts execute inside the bwrap sandbox. They are injected as
Python source strings and run via run_python_sandboxed().

The combined runner (COMBINED_RUNNER) handles:
1. Loading Python and Cython modules
2. Running test assertions (py.func() == cy.func() style)
3. Running benchmarks
4. Outputting structured JSON that the training environment parses

Output format is designed for LLM self-healing: every failure includes
actionable diagnostic information.
"""

# ---------------------------------------------------------------------------
# Combined test + benchmark runner for training environment
# ---------------------------------------------------------------------------

COMBINED_RUNNER = """\
import sys, json, os, importlib, importlib.util, time, types, signal, re, contextlib

# --- Module loading ---

def _load_module_from_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    mod_dir = os.path.dirname(os.path.abspath(path))
    if mod_dir not in sys.path:
        sys.path.insert(0, mod_dir)
    spec.loader.exec_module(mod)
    return mod

def _exec_as_module(code, name="dynamic_module"):
    mod = types.ModuleType(name)
    exec(code, mod.__dict__)
    return mod

# --- Timeout handling ---

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

# --- Test runner ---

def run_tests(py_mod, cy_mod, test_code):
    namespace = {"py": py_mod, "cy": cy_mod}
    passed = 0
    total = 0
    failures = []

    for line in test_code.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        try:
            with _alarm(ASSERTION_TIMEOUT):
                if "==" in line:
                    total += 1
                    result = eval(line, namespace)
                    if result:
                        passed += 1
                    else:
                        parts = line.split("==", 1)
                        try:
                            left = eval(parts[0].strip(), namespace)
                            right = eval(parts[1].strip(), namespace)
                            failures.append(
                                f"FAIL: {line}\\n  left:  {repr(left)[:200]}\\n  right: {repr(right)[:200]}"
                            )
                        except Exception:
                            failures.append(f"FAIL: {line}")
                else:
                    exec(line, namespace)
        except _TimeoutError:
            if "==" in line:
                total += 1
                failures.append(f"TIMEOUT: {line} (>{ASSERTION_TIMEOUT}s)")
            else:
                failures.append(f"TIMEOUT (setup): {line}")
        except Exception as e:
            if "==" in line:
                total += 1
                failures.append(f"ERROR: {line}\\n  {type(e).__name__}: {e}")
            else:
                failures.append(f"ERROR (setup): {line}\\n  {type(e).__name__}: {e}")

    return {"passed": passed, "total": total, "failures": failures}

# --- Benchmark ---

def run_benchmark(py_mod, cy_mod, test_code):
    for line in test_code.strip().splitlines():
        line = line.strip()
        m = re.match(r"py\\.(\\w+)\\((.+?)\\)\\s*==\\s*cy\\.\\w+\\(", line)
        if m:
            func_name = m.group(1)
            py_func = getattr(py_mod, func_name, None)
            cy_func = getattr(cy_mod, func_name, None)
            if py_func and cy_func:
                args_str = m.group(2)
                try:
                    args = eval(f"({args_str},)")
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

# --- Main ---

config = json.loads(open(os.path.join(os.path.dirname(__file__), "_combined_config.json")).read())

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
    cy_mod = _load_module_from_path(cython_module_path, module_name)
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
"""
