"""
Memory safety checking via AddressSanitizer (ASan).

Compiles Cython code with -fsanitize=address and runs it with small inputs
to detect memory errors: leaks, use-after-free, buffer overflows, double-free.
Uses PYTHONMALLOC=malloc to avoid false positives from CPython's arena allocator.
"""

import logging
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

ASAN_COMPILE_FLAGS = [
    "-fsanitize=address",
    "-fno-omit-frame-pointer",
    "-g",
    "-O1",
]

ASAN_LINK_FLAGS = [
    "-fsanitize=address",
]

# ASan environment: force system malloc so CPython allocator noise is eliminated
ASAN_ENV = {
    "PYTHONMALLOC": "malloc",
    "ASAN_OPTIONS": "detect_leaks=1:fast_unwind_on_malloc=0:strict_string_checks=1",
    # Preload ASan so it's available to the extension module
    "LD_PRELOAD": "",  # populated dynamically
}

SETUP_PY_TEMPLATE = """\
from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        [Extension("{module_name}", ["{module_name}.pyx"],
                   extra_compile_args={extra_compile_args},
                   extra_link_args={extra_link_args})],
        compiler_directives={{
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
        }},
        annotate=False,
    ),
)
"""

# Pattern to match ASan error reports
ASAN_ERROR_RE = re.compile(r"==\d+==ERROR: (AddressSanitizer|LeakSanitizer): (.+?)(?:\n|$)")
ASAN_SUMMARY_RE = re.compile(r"SUMMARY: (AddressSanitizer|LeakSanitizer): (\d+) byte\(s\) (.+)")
ASAN_LEAK_RE = re.compile(r"(Direct|Indirect) leak of (\d+) byte\(s\) in (\d+) object\(s\)")


@dataclass
class MemorySafetyResult:
    success: bool = True
    score: float = 1.0
    errors: list[str] = field(default_factory=list)
    leak_bytes: int = 0
    error_count: int = 0
    error_types: list[str] = field(default_factory=list)
    raw_output: str = ""


def _find_asan_lib() -> str:
    """Find the ASan shared library for LD_PRELOAD."""
    try:
        result = subprocess.run(
            ["gcc", "-print-file-name=libasan.so"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        path = result.stdout.strip()
        if path and Path(path).exists():
            return path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: search common locations
    for pattern in [
        "/usr/lib/x86_64-linux-gnu/libasan.so*",
        "/usr/lib64/libasan.so*",
        "/usr/lib/libasan.so*",
    ]:
        import glob

        matches = sorted(glob.glob(pattern), reverse=True)
        if matches:
            return matches[0]

    return ""


def check_memory_safety(
    cython_code: str,
    func_name: str,
    test_args: tuple = (),
    module_name: str = "asan_module",
    timeout: int = 60,
    extra_compile_args: list | None = None,
) -> MemorySafetyResult:
    """
    Compile Cython code with ASan and run it to detect memory errors.

    Args:
        cython_code: The .pyx source code as a string.
        func_name: Name of the function to call.
        test_args: Arguments to pass to the function (use small values).
        module_name: Name for the compiled module.
        timeout: Subprocess timeout in seconds.
        extra_compile_args: Additional compiler flags.

    Returns:
        MemorySafetyResult with score (1.0 = clean, 0.0 = errors found).
    """
    result = MemorySafetyResult()

    with tempfile.TemporaryDirectory(prefix="cnake_asan_") as tmpdir:
        try:
            result = _run_asan_in_dir(
                tmpdir,
                result,
                cython_code,
                func_name,
                test_args,
                module_name,
                timeout,
                extra_compile_args,
            )
        except subprocess.TimeoutExpired:
            result.success = False
            result.errors = [f"ASan check timed out after {timeout}s"]
            result.score = 1.0  # Don't penalize timeouts
        except Exception as e:
            result.success = False
            result.errors = [str(e)]
            result.score = 1.0  # Don't penalize infrastructure failures

    return result


def _run_asan_in_dir(
    tmpdir: str,
    result: MemorySafetyResult,
    cython_code: str,
    func_name: str,
    test_args: tuple,
    module_name: str,
    timeout: int,
    extra_compile_args: list | None,
) -> MemorySafetyResult:
    """Inner ASan logic, separated so TemporaryDirectory handles cleanup."""
    # Write .pyx file (strip benchmark decorators)
    clean_code = _strip_decorators(cython_code)
    pyx_path = os.path.join(tmpdir, f"{module_name}.pyx")
    with open(pyx_path, "w") as f:
        f.write(clean_code)

    # Build compile flags
    compile_flags = list(ASAN_COMPILE_FLAGS)
    if extra_compile_args:
        compile_flags.extend(extra_compile_args)

    # Write setup.py
    setup_content = SETUP_PY_TEMPLATE.format(
        module_name=module_name,
        extra_compile_args=compile_flags,
        extra_link_args=ASAN_LINK_FLAGS,
    )
    with open(os.path.join(tmpdir, "setup.py"), "w") as f:
        f.write(setup_content)

    # Compile with ASan flags (sandboxed, but WITHOUT ASan runtime env)
    from cnake_charmer.eval.sandbox import asan_config, compile_config, run_sandboxed

    compile_sandbox_cfg = compile_config(
        wall_time_limit_s=timeout + 30,
        cpu_time_limit_s=timeout,
        writable_paths=(tmpdir,),
    )
    compile_sandbox = run_sandboxed(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        config=compile_sandbox_cfg,
        cwd=tmpdir,
    )

    if compile_sandbox.returncode != 0:
        result.success = False
        result.errors = [f"ASan build failed: {compile_sandbox.stderr[:500]}"]
        result.score = 1.0  # Don't penalize build failures (separate concern)
        return result

    # Build the test runner script
    args_repr = repr(test_args)
    runner_code = f"""\
import sys
sys.path.insert(0, '.')
from {module_name} import {func_name}
result = {func_name}(*{args_repr})
print(f"OK: {{result}}")
"""
    runner_path = os.path.join(tmpdir, "run_asan_test.py")
    with open(runner_path, "w") as f:
        f.write(runner_code)

    # Run the test with ASan in sandbox
    asan_lib = _find_asan_lib()
    asan_opts = "detect_leaks=1:fast_unwind_on_malloc=0:print_legend=0:log_path=stderr"

    # No LSan suppressions — we filter leaks by module in _parse_asan_output
    supp_path = os.path.join(tmpdir, "lsan.supp")
    with open(supp_path, "w") as f:
        f.write("")  # empty file

    asan_extra_env = {
        "PYTHONMALLOC": "malloc",
        "ASAN_OPTIONS": asan_opts,
        "LSAN_OPTIONS": "suppressions=" + supp_path,
    }
    if asan_lib:
        asan_extra_env["LD_PRELOAD"] = asan_lib

    run_sandbox_cfg = asan_config(
        wall_time_limit_s=timeout + 30,
        cpu_time_limit_s=timeout,
        writable_paths=(tmpdir,),
        extra_env=asan_extra_env,
    )
    assert run_sandbox_cfg.memory_limit_mb == 0, (
        f"ASan requires unlimited virtual address space for shadow memory, "
        f"but memory_limit_mb={run_sandbox_cfg.memory_limit_mb}"
    )
    run_sandbox = run_sandboxed(
        [sys.executable, "run_asan_test.py"],
        config=run_sandbox_cfg,
        cwd=tmpdir,
    )

    result.raw_output = run_sandbox.stderr

    # Parse ASan output — only count errors from user function
    _parse_asan_output(
        run_sandbox.stderr,
        result,
        module_name=module_name,
        func_name=func_name,
    )

    # Check if the function actually ran successfully
    if "OK:" not in run_sandbox.stdout and run_sandbox.returncode != 0 and not result.errors:
        result.errors.append(f"Function crashed (exit code {run_sandbox.returncode})")
        result.error_count += 1
        result.score = 0.0

    return result


def _parse_asan_output(
    stderr: str,
    result: MemorySafetyResult,
    module_name: str = "asan_module",
    func_name: str = "",
) -> None:
    """Parse ASan/LSan output, only counting errors from user code.

    Filters by checking if the user's function name appears in the stack trace.
    Cython generates names like __pyx_pw_11module_1func_name or __pyx_f_11module_func_name.
    CPython internal leaks (module init, type caches) are ignored.

    ASan errors (buffer overflow, use-after-free) in the .so are always counted.
    """
    if not stderr:
        return

    so_pattern = f"{module_name}.cpython"

    # Build patterns that identify user function code (not module init)
    # Cython generates: __pyx_pw_<len><module>_<N><func_name> (Python wrapper)
    #                    __pyx_f_<len><module>_<func_name> (C function)
    user_patterns = []
    if func_name:
        user_patterns.append(func_name)
    user_patterns.append(so_pattern)

    # Patterns to identify our compiled module in stack traces
    # ASan shows: __pyx_pf_11asan_module_func_name (in the .c or .so file)
    module_patterns = [so_pattern, f"{module_name}.c"]

    def _is_user_code(block: str) -> bool:
        """Check if a stack trace block originates from user function code."""
        has_module = any(p in block for p in module_patterns)
        if not has_module:
            return False
        if not func_name:
            return True
        return func_name in block

    def _is_module_code(block: str) -> bool:
        """Check if a stack trace involves our module at all (for non-leak errors)."""
        return any(p in block for p in module_patterns)

    # Split into blocks by "Direct leak" / "Indirect leak" / error headers
    # ASan output is structured as error header + stack trace blocks
    blocks = re.split(r"\n(?=Direct leak|Indirect leak|==\d+==ERROR)", stderr)

    for block in blocks:
        # Non-leak ASan errors (buffer overflow, use-after-free, etc.)
        error_match = ASAN_ERROR_RE.search(block)
        if error_match:
            tool = error_match.group(1)
            error_type = error_match.group(2)
            if "leak" in error_type.lower():
                continue
            if _is_module_code(block):
                result.errors.append(f"{tool}: {error_type}")
                result.error_count += 1
                if error_type not in result.error_types:
                    result.error_types.append(error_type)

        # Leak reports — only count if from user function
        leak_match = ASAN_LEAK_RE.search(block)
        if leak_match and _is_user_code(block):
            leak_kind = leak_match.group(1)
            leak_bytes = int(leak_match.group(2))
            result.leak_bytes += leak_bytes
            result.error_count += 1
            error_msg = f"{leak_kind} leak: {leak_bytes} bytes"
            result.errors.append(error_msg)
            if "leak" not in result.error_types:
                result.error_types.append("leak")

    # Binary score: any user-code error = 0
    result.score = 0.0 if result.error_count > 0 else 1.0


def _strip_decorators(code: str) -> str:
    """Strip benchmark decorators from Cython code for ASan testing."""
    lines = code.splitlines()
    filtered = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("@cython_benchmark") or stripped.startswith("@python_benchmark"):
            continue
        if "from cnake_data.benchmarks import" in stripped:
            continue
        filtered.append(line)
    return "\n".join(filtered)


def memory_safety_reward(
    cython_code: str,
    func_name: str,
    test_args: tuple = (),
    **kwargs,
) -> float:
    """
    Return memory safety score (0.0 or 1.0).

    1.0 = no memory errors detected, 0.0 = errors found.
    """
    result = check_memory_safety(cython_code, func_name, test_args)
    return result.score
