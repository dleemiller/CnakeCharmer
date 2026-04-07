"""
Tests for the bwrap sandbox module.

Covers:
- Basic execution
- Filesystem isolation (can't read /home, .ssh, .env, repo files)
- Network isolation (can't connect to external hosts)
- Resource limits (memory, CPU, process count, file size)
- Wall-clock timeout (uncatchable by malicious code)
- OOM detection
- Compilation through sandbox
- Full pipeline integration
"""

import json
import sys

from cnake_charmer.eval.sandbox import (
    SandboxConfig,
    execute_config,
    run_sandboxed,
)

_PY = sys.executable


def _run_script(script: str, config: SandboxConfig | None = None):
    """Convenience: run a Python snippet in the sandbox."""
    kw = {"config": config} if config else {}
    return run_sandboxed([_PY, "-c", script], **kw)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_to_rlimits_dict_keys():
    """to_rlimits_dict() returns the exact keys apply_rlimits() expects."""
    cfg = SandboxConfig(
        memory_limit_mb=1024, cpu_time_limit_s=10, max_processes=16, max_file_size_mb=32
    )
    d = cfg.to_rlimits_dict()
    assert d == {
        "memory_mb": 1024,
        "cpu_time_s": 10,
        "max_processes": 16,
        "max_file_size_mb": 32,
    }


# ---------------------------------------------------------------------------
# Basic execution
# ---------------------------------------------------------------------------


def test_basic_execution():
    """Sandbox can run a simple Python command and capture output."""
    result = run_sandboxed([_PY, "-c", 'print("hello sandbox")'])
    assert result.returncode == 0
    assert "hello sandbox" in result.stdout
    assert not result.timed_out
    assert not result.oom_killed


def test_run_script():
    """_run_script convenience wrapper works."""
    result = _run_script('print("script works")')
    assert result.returncode == 0
    assert "script works" in result.stdout


def test_imports_in_sandbox():
    """Standard imports (numpy, Cython) work inside sandbox."""
    result = _run_script(
        "import numpy as np; import Cython; "
        'print(f"numpy={np.__version__}"); '
        'print(f"cython={Cython.__version__}")'
    )
    assert result.returncode == 0
    assert "numpy=" in result.stdout
    assert "cython=" in result.stdout


# ---------------------------------------------------------------------------
# Filesystem isolation
# ---------------------------------------------------------------------------


def test_cannot_read_home_files():
    """Sandbox cannot access real files in /home."""
    result = _run_script(
        "import os, json; "
        "results = {}; "
        "[results.__setitem__(p, 'ACCESSIBLE' if open(p).read(1) else 'ACCESSIBLE') "
        "if False else None for p in []]; "  # dummy to avoid syntax issues
        "paths = [os.path.expanduser('~/.bashrc'), os.path.expanduser('~/.ssh'), '/etc/shadow']; "
        "r = {}; "
        "[r.__setitem__(p, 'BLOCKED') for p in paths]; "
        "exec('\\n'.join("
        "['try:\\n with open(p) as f: f.read(1)\\n r[p]=\"ACCESSIBLE\"\\n'"
        " 'except (FileNotFoundError, PermissionError): pass' for p in paths]"
        "), {'r': r, 'paths': paths}); "  # noqa
        "print(json.dumps(r))"
    )
    # Simpler approach: use a script file
    result = run_sandboxed(
        [
            _PY,
            "-c",
            """
import os, json
results = {}
for path in [os.path.expanduser("~/.bashrc"), os.path.expanduser("~/.ssh"), "/etc/shadow"]:
    try:
        with open(path) as f:
            f.read(1)
        results[path] = "ACCESSIBLE"
    except (FileNotFoundError, PermissionError):
        results[path] = "BLOCKED"
print(json.dumps(results))
""",
        ]
    )
    assert result.returncode == 0
    data = json.loads(result.stdout.strip())
    for path, status in data.items():
        assert status == "BLOCKED", f"{path} should be blocked but is {status}"


def test_cannot_read_repo_files():
    """Sandbox cannot access the git repository source code."""
    result = run_sandboxed(
        [
            _PY,
            "-c",
            """
import os
repo = "/home/lee/Documents/code/CnakeCharmer"
try:
    contents = os.listdir(repo)
    non_venv = [c for c in contents if c != ".venv"]
    if non_venv:
        print(f"FAIL: repo files visible: {non_venv}")
    else:
        print("PASS: only .venv visible")
except FileNotFoundError:
    print("PASS: repo not visible")
""",
        ]
    )
    assert result.returncode == 0
    assert "PASS" in result.stdout


def test_venv_is_readonly():
    """Sandbox cannot write to the venv directory."""
    result = run_sandboxed(
        [
            _PY,
            "-c",
            """
import sys, os
venv = sys.prefix
try:
    with open(os.path.join(venv, "test_write"), "w") as f:
        f.write("pwned")
    print("FAIL: wrote to venv")
except (PermissionError, OSError):
    print("PASS: venv is read-only")
""",
        ]
    )
    assert result.returncode == 0
    assert "PASS" in result.stdout


# ---------------------------------------------------------------------------
# Network isolation
# ---------------------------------------------------------------------------


def test_network_blocked():
    """Sandbox cannot make network connections."""
    result = run_sandboxed(
        [
            _PY,
            "-c",
            """
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    s.connect(("8.8.8.8", 53))
    print("FAIL: network accessible")
except OSError:
    print("PASS: network blocked")
""",
        ]
    )
    assert result.returncode == 0
    assert "PASS" in result.stdout


# ---------------------------------------------------------------------------
# Resource limits
# ---------------------------------------------------------------------------


def test_resource_limits_applied():
    """Resource limits are set correctly inside the sandbox.

    RLIMIT_NPROC is excluded: it can't be set on the bwrap process
    (clone() for namespace creation counts against it).  The PID
    namespace provides equivalent fork-bomb protection.  Runner
    scripts apply RLIMIT_NPROC themselves via _common.apply_rlimits().
    """
    cfg = execute_config()
    result = _run_script(
        "import resource; "
        "print(f'RLIMIT_AS={resource.getrlimit(resource.RLIMIT_AS)[0]}'); "
        "print(f'RLIMIT_CPU={resource.getrlimit(resource.RLIMIT_CPU)[0]}'); "
        "print(f'RLIMIT_FSIZE={resource.getrlimit(resource.RLIMIT_FSIZE)[0]}')",
        config=cfg,
    )
    assert result.returncode == 0
    limits = {}
    for line in result.stdout.strip().splitlines():
        key, val = line.split("=")
        limits[key] = int(val)
    assert limits["RLIMIT_AS"] == cfg.memory_limit_mb * 1024 * 1024
    assert limits["RLIMIT_CPU"] == cfg.cpu_time_limit_s
    assert limits["RLIMIT_FSIZE"] == cfg.max_file_size_mb * 1024 * 1024


def test_wall_clock_timeout():
    """Wall-clock timeout kills the process reliably."""
    cfg = SandboxConfig(wall_time_limit_s=2, cpu_time_limit_s=5)
    result = _run_script("import time; time.sleep(60)", config=cfg)
    assert result.timed_out
    assert result.wall_time_s < 5


def test_wall_clock_timeout_uncatchable():
    """Timeout cannot be caught by signal handlers inside sandbox."""
    cfg = SandboxConfig(wall_time_limit_s=2, cpu_time_limit_s=10)
    result = run_sandboxed(
        [
            _PY,
            "-c",
            """
import signal, time
signal.signal(signal.SIGTERM, lambda *a: None)
signal.signal(signal.SIGINT, lambda *a: None)
time.sleep(60)
""",
        ],
        config=cfg,
    )
    assert result.timed_out
    assert result.wall_time_s < 5


def test_fork_bomb_contained():
    """Fork bombs are contained by the sandbox.

    In bwrap mode, RLIMIT_NPROC can't be set from outside because it
    counts all host-UID processes.  The PID namespace contains forked
    processes.  Runner scripts apply RLIMIT_NPROC inside the sandbox
    for actual code execution (belt-and-suspenders).

    This test verifies that a fork bomb completes without crashing or
    hanging the sandbox — the wall-clock timeout is the backstop.
    """
    cfg = SandboxConfig(wall_time_limit_s=5, cpu_time_limit_s=5, max_processes=8)
    result = run_sandboxed(
        [
            _PY,
            "-c",
            """
import os
count = 0
try:
    for _ in range(100):
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        count += 1
except OSError as e:
    print(f"fork limited at {count}: {e}")
else:
    print(f"forked {count} times")
""",
        ],
        config=cfg,
    )
    # Either RLIMIT_NPROC kicked in (fallback path) or it was contained
    # by the PID namespace (bwrap path).  Either way, we completed.
    assert result.returncode == 0 or result.timed_out


# ---------------------------------------------------------------------------
# Compilation through sandbox
# ---------------------------------------------------------------------------


def test_compilation_sandboxed():
    """Cython compilation works through the sandbox."""
    from cnake_charmer.eval.compiler import cleanup_build, compile_cython

    code = "def add(int a, int b):\n    return a + b\n"
    result = compile_cython(code, keep_build=True)
    assert result.success
    assert result.module_path is not None
    assert result.module_path.endswith(".so")
    cleanup_build(result)


def test_compilation_with_flags():
    """Compilation with extra flags (SIMD) works in sandbox."""
    from cnake_charmer.eval.compiler import cleanup_build, compile_cython

    code = "def mul(double a, double b):\n    return a * b\n"
    result = compile_cython(code, extra_compile_args=["-mavx2", "-O3"])
    assert result.success
    cleanup_build(result)


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------


def test_full_pipeline_sandboxed():
    """Full composite_reward pipeline runs entirely in sandbox."""
    from cnake_charmer.eval.pipeline import composite_reward

    python_code = (
        "def fibonacci(n):\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n):\n"
        "        a, b = b, a + b\n"
        "    return a\n"
    )
    cython_code = (
        "def fibonacci(int n):\n"
        "    cdef int a = 0, b = 1, i\n"
        "    for i in range(n):\n"
        "        a, b = b, a + b\n"
        "    return a\n"
    )
    test_cases = [((0,),), ((1,),), ((5,),), ((10,),)]

    scores = composite_reward(
        cython_code=cython_code,
        python_code=python_code,
        func_name="fibonacci",
        test_cases=test_cases,
        benchmark_args=(30,),
        benchmark_runs=3,
    )

    assert scores["compiled"]
    assert scores["correctness"] == 1.0
    assert scores["speedup"] > 1.0
    assert scores["total"] > 0.5


def test_training_environment_sandboxed():
    """Training environment works with sandboxed execution."""
    from cnake_charmer.training.environment import CythonToolEnvironment

    env = CythonToolEnvironment()
    env.reset()

    output = env.evaluate_cython(
        code=(
            "def fibonacci(int n):\n"
            "    cdef int a = 0, b = 1, i\n"
            "    for i in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a\n"
        ),
        python_code=(
            "def fibonacci(n):\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n): a, b = b, a + b\n"
            "    return a\n"
        ),
        test_code="py.fibonacci(0) == cy.fibonacci(0)\npy.fibonacci(10) == cy.fibonacci(10)",
    )

    assert "## Compilation" in output
    assert "Compilation successful" in output
    assert "## Tests" in output
    assert "2/2 passed" in output
    assert env.step_scores[-1]["compiled"]
    assert env.step_scores[-1]["correctness"] == 1.0


def test_mcp_evaluate_cython():
    """MCP evaluate_cython tool works through sandbox."""
    from cnake_charmer.mcp_server import evaluate_cython

    output = evaluate_cython(
        code="def double(int n):\n    return n * 2\n",
        python_code="def double(n):\n    return n * 2\n",
        test_code="py.double(0) == cy.double(0)\npy.double(5) == cy.double(5)",
    )

    assert "## Compilation" in output
    assert "## Tests" in output
    assert "2/2 passed" in output


# ---------------------------------------------------------------------------
# Memory safety (ASan)
# ---------------------------------------------------------------------------


def test_asan_clean_code():
    """ASan check passes for memory-safe Cython code."""
    from cnake_charmer.eval.memory_safety import check_memory_safety

    code = "def add(int a, int b):\n    return a + b\n"
    result = check_memory_safety(code, "add", test_args=(3, 4))
    assert result.score == 1.0, f"Expected clean ASan run, got: {result.errors}"


def test_asan_in_composite_reward():
    """Memory safety component scores > 0 in composite_reward."""
    from cnake_charmer.eval.pipeline import composite_reward

    python_code = "def double(n):\n    return n * 2\n"
    cython_code = "def double(int n):\n    return n * 2\n"

    scores = composite_reward(
        cython_code=cython_code,
        python_code=python_code,
        func_name="double",
        test_cases=[((0,),), ((5,),), ((10,),)],
        benchmark_args=(1000,),
        benchmark_runs=3,
    )
    assert scores["memory_safety"] > 0, (
        f"memory_safety=0 means ASan failed to run: {scores.get('memory_safety_errors')}"
    )


# ---------------------------------------------------------------------------
# Adversarial: dangerous code patterns
# ---------------------------------------------------------------------------


def test_dangerous_import_os_system():
    """Code trying os.system is contained by sandbox."""
    result = run_sandboxed(
        [
            _PY,
            "-c",
            """
import os, os.path
os.system("touch /usr/PWNED 2>/dev/null")
if os.path.exists("/usr/PWNED"):
    print("FAIL: file created")
else:
    print("PASS: write blocked")
""",
        ]
    )
    assert "PASS" in result.stdout


def test_dangerous_subprocess():
    """Code trying subprocess.run is contained by sandbox."""
    result = run_sandboxed(
        [
            _PY,
            "-c",
            """
import subprocess
try:
    r = subprocess.run(["cat", "/etc/shadow"], capture_output=True, text=True)
    if r.returncode == 0 and r.stdout:
        print("FAIL: read /etc/shadow")
    else:
        print("PASS: access denied")
except Exception as e:
    print(f"PASS: {e}")
""",
        ]
    )
    assert "PASS" in result.stdout
