"""
Bubblewrap (bwrap) sandbox for untrusted code execution.

All compilation and code execution in the eval pipeline flows through
run_sandboxed(), which wraps commands in a bwrap container with:
- Filesystem isolation (read-only system libs, no /home access)
- Network isolation (unshared network namespace)
- Resource limits (memory, CPU, processes, file size via prlimit)
- Wall-clock watchdog (uncatchable SIGKILL on timeout)
- PID namespace (prevents signaling host processes)

Three profiles (compile, execute, asan) provide the right mount
configuration for each use case.

Falls back to prlimit-only if bwrap is unavailable.
"""

import contextlib
import json
import logging
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from functools import lru_cache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_BWRAP = shutil.which("bwrap")
_SANDBOX_ENABLED = os.environ.get("CNAKE_SANDBOX_ENABLED", "1" if _BWRAP else "0") == "1"

# Environment-based overrides for resource limits
_MEMORY_MB = int(os.environ.get("CNAKE_SANDBOX_MEMORY_MB", "0")) or None
_WALL_TIMEOUT = int(os.environ.get("CNAKE_SANDBOX_WALL_TIMEOUT_S", "0")) or None


@dataclass(frozen=True)
class SandboxConfig:
    """Immutable configuration for a sandbox invocation."""

    memory_limit_mb: int = 2048
    cpu_time_limit_s: int = 120
    wall_time_limit_s: int = 150
    max_processes: int = 32
    max_file_size_mb: int = 256
    tmpfs_size_mb: int = 512
    network: bool = False
    writable_paths: tuple[str, ...] = ()
    extra_ro_binds: tuple[str, ...] = ()
    extra_env: dict[str, str] = field(default_factory=dict)


@dataclass
class SandboxResult:
    """Structured result from a sandboxed invocation."""

    returncode: int
    stdout: str
    stderr: str
    wall_time_s: float
    timed_out: bool
    oom_killed: bool


# ---------------------------------------------------------------------------
# Profile factories
# ---------------------------------------------------------------------------


def compile_config(**overrides) -> SandboxConfig:
    """Profile for Cython compilation. Needs gcc, Python headers, Cython."""
    defaults = dict(
        cpu_time_limit_s=120,
        wall_time_limit_s=150,
        memory_limit_mb=2048,
        max_file_size_mb=256,
        tmpfs_size_mb=512,
    )
    defaults.update(overrides)
    return SandboxConfig(**defaults)


def execute_config(**overrides) -> SandboxConfig:
    """Profile for running compiled code. Read-only — no writable bind mounts."""
    defaults = dict(
        cpu_time_limit_s=30,
        wall_time_limit_s=45,
        memory_limit_mb=2048,
        max_file_size_mb=64,
        tmpfs_size_mb=128,
    )
    defaults.update(overrides)
    return SandboxConfig(**defaults)


def asan_config(**overrides) -> SandboxConfig:
    """Profile for AddressSanitizer. Needs ASan lib + compile toolchain."""
    from cnake_charmer.eval.memory_safety import _find_asan_lib

    asan_lib = _find_asan_lib()
    extra_ro = tuple(overrides.pop("extra_ro_binds", ()))
    if asan_lib:
        extra_ro = extra_ro + (asan_lib,)

    extra_env = dict(overrides.pop("extra_env", {}))
    extra_env.setdefault("PYTHONMALLOC", "malloc")
    asan_opts = "detect_leaks=1:fast_unwind_on_malloc=0:print_legend=0:log_path=stderr"
    extra_env.setdefault("ASAN_OPTIONS", asan_opts)
    if asan_lib:
        extra_env.setdefault("LD_PRELOAD", asan_lib)

    defaults = dict(
        cpu_time_limit_s=60,
        wall_time_limit_s=90,
        memory_limit_mb=2048,
        max_file_size_mb=256,
        tmpfs_size_mb=512,
        extra_ro_binds=extra_ro,
        extra_env=extra_env,
    )
    defaults.update(overrides)
    return SandboxConfig(**defaults)


# ---------------------------------------------------------------------------
# Path discovery (cached)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _discover_paths() -> dict[str, str]:
    """Discover and cache system paths needed for bwrap mounts."""
    venv = sys.prefix
    python = sys.executable
    return {
        "venv": venv,
        "python": python,
    }


# ---------------------------------------------------------------------------
# Bwrap command construction
# ---------------------------------------------------------------------------


def _build_bwrap_cmd(
    command: list[str],
    config: SandboxConfig,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> list[str]:
    """Build the full bwrap command line."""
    paths = _discover_paths()
    venv = paths["venv"]

    cmd = [
        _BWRAP,
        # Namespace isolation
        "--unshare-user",
        "--unshare-ipc",
        "--unshare-pid",
        "--unshare-uts",
        "--unshare-cgroup-try",
        "--disable-userns",  # prevent nested namespace escape
        "--die-with-parent",  # cleanup if parent dies
        "--new-session",  # prevent tty injection (CVE-2017-5226)
    ]

    # Network: unshare by default, share only if explicitly requested
    if not config.network:
        cmd.append("--unshare-net")

    # Core filesystem
    cmd.extend(
        [
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            # System libraries (covers /lib, /lib64 which are symlinks to /usr/*)
            "--ro-bind",
            "/usr",
            "/usr",
            "--symlink",
            "usr/lib",
            "/lib",
            "--symlink",
            "usr/lib64",
            "/lib64",
            # Linker cache
            "--ro-bind",
            "/etc/ld.so.cache",
            "/etc/ld.so.cache",
            "--ro-bind-try",
            "/etc/alternatives",
            "/etc/alternatives",
        ]
    )

    # Writable tmpfs areas
    cmd.extend(["--size", str(config.tmpfs_size_mb * 1024 * 1024), "--tmpfs", "/tmp"])
    # Ephemeral tmpfs over /home so parent dirs for venv mount are writable
    # only in RAM, not on real filesystem
    cmd.extend(["--size", str(10 * 1024 * 1024), "--tmpfs", "/home"])

    # Venv (read-only)
    cmd.extend(["--ro-bind", venv, venv])

    # Extra read-only binds
    for path in config.extra_ro_binds:
        if path and os.path.exists(path):
            cmd.extend(["--ro-bind", path, path])

    # Writable bind mounts (e.g., compilation temp dir)
    for path in config.writable_paths:
        cmd.extend(["--bind", path, path])

    # Environment: start clean, set only what's needed
    cmd.append("--clearenv")
    cmd.extend(["--setenv", "HOME", "/tmp"])
    cmd.extend(["--setenv", "PATH", f"{venv}/bin:/usr/bin:/usr/local/bin"])
    cmd.extend(["--setenv", "VIRTUAL_ENV", venv])
    cmd.extend(["--setenv", "PYTHONDONTWRITEBYTECODE", "1"])
    # Avoid .pyc write errors in read-only locations
    cmd.extend(["--setenv", "PYTHONPYCACHEPREFIX", "/tmp/__pycache__"])

    # Profile-specific env vars
    for key, value in config.extra_env.items():
        cmd.extend(["--setenv", key, value])

    # Caller-specified env vars
    if env:
        for key, value in env.items():
            cmd.extend(["--setenv", key, value])

    # Working directory
    if cwd:
        cmd.extend(["--chdir", cwd])

    # The actual command to run
    cmd.extend(command)
    return cmd


# ---------------------------------------------------------------------------
# Resource limits (prlimit via preexec_fn)
# ---------------------------------------------------------------------------


def _make_preexec_fn(config: SandboxConfig, *, apply_rlimits: bool = True):
    """Create a preexec_fn that sets resource limits before exec.

    When wrapping bwrap, set apply_rlimits=False — bwrap needs full
    process/memory allowance to set up namespaces. Resource limits are
    applied inside the sandbox via _rlimit_preamble() instead.
    """
    mem_bytes = config.memory_limit_mb * 1024 * 1024
    fsize_bytes = config.max_file_size_mb * 1024 * 1024

    def _set_limits():
        # New process group so killpg works for wall-clock watchdog
        os.setpgrp()
        if apply_rlimits:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            resource.setrlimit(
                resource.RLIMIT_CPU, (config.cpu_time_limit_s, config.cpu_time_limit_s)
            )
            resource.setrlimit(resource.RLIMIT_NPROC, (config.max_processes, config.max_processes))
            resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_bytes, fsize_bytes))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return _set_limits


def _rlimit_preamble(config: SandboxConfig) -> str:
    """Python code snippet that sets resource limits. Injected into sandbox scripts."""
    mem_bytes = config.memory_limit_mb * 1024 * 1024
    fsize_bytes = config.max_file_size_mb * 1024 * 1024
    return f"""\
import resource as _r
_r.setrlimit(_r.RLIMIT_AS, ({mem_bytes}, {mem_bytes}))
_r.setrlimit(_r.RLIMIT_CPU, ({config.cpu_time_limit_s}, {config.cpu_time_limit_s}))
_r.setrlimit(_r.RLIMIT_NPROC, ({config.max_processes}, {config.max_processes}))
_r.setrlimit(_r.RLIMIT_FSIZE, ({fsize_bytes}, {fsize_bytes}))
_r.setrlimit(_r.RLIMIT_CORE, (0, 0))
del _r
"""


# ---------------------------------------------------------------------------
# Core execution
# ---------------------------------------------------------------------------


def _log_event(event: str, profile: str = "", **extra):
    """Emit structured log event."""
    data = {"event": event, "profile": profile, **extra}
    logger.info(json.dumps(data, default=str))


_DEFAULT_CONFIG = SandboxConfig()


def _kill_proc_tree(proc: subprocess.Popen) -> None:
    """Kill a process and its entire process group. Best-effort, never raises."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except OSError:
        # Process already dead, wrong pgid, or no permission — fall back
        with contextlib.suppress(OSError):
            proc.kill()


def run_sandboxed(
    command: list[str],
    *,
    config: SandboxConfig = _DEFAULT_CONFIG,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> SandboxResult:
    """Execute a command inside a bwrap sandbox with resource limits.

    This is the single choke point for all sandboxed execution.
    Falls back to prlimit-only if bwrap is unavailable.
    """
    if not _SANDBOX_ENABLED or not _BWRAP:
        return _run_prlimit_only(command, config=config, cwd=cwd, env=env)

    bwrap_cmd = _build_bwrap_cmd(command, config, cwd=cwd, env=env)

    cmd_summary = " ".join(command)[:200]
    _log_event("sandbox.start", command_summary=cmd_summary)
    start = time.monotonic()

    timed_out = False
    oom_killed = False

    try:
        proc = subprocess.Popen(
            bwrap_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Don't apply rlimits to bwrap itself — it needs full resources
            # to create namespaces. Limits are applied inside the sandbox.
            preexec_fn=_make_preexec_fn(config, apply_rlimits=False),
        )

        try:
            stdout_bytes, stderr_bytes = proc.communicate(timeout=config.wall_time_limit_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_proc_tree(proc)
            try:
                stdout_bytes, stderr_bytes = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                # Process is truly stuck — reap it to avoid zombie
                proc.kill()
                proc.wait()
                stdout_bytes, stderr_bytes = b"", b""

        wall_time = time.monotonic() - start
        returncode = proc.returncode

        # Detect OOM kill (signal 9 = returncode -9 or 137)
        oom_killed = returncode in (-9, 137)

        result = SandboxResult(
            returncode=returncode,
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            wall_time_s=wall_time,
            timed_out=timed_out,
            oom_killed=oom_killed,
        )

        _log_event(
            "sandbox.complete",
            wall_time_s=round(wall_time, 3),
            returncode=returncode,
            timed_out=timed_out,
            oom_killed=oom_killed,
            command_summary=cmd_summary,
        )
        return result

    except Exception as e:
        wall_time = time.monotonic() - start
        _log_event("sandbox.error", error=str(e), wall_time_s=round(wall_time, 3))
        return SandboxResult(
            returncode=-1,
            stdout="",
            stderr=str(e),
            wall_time_s=wall_time,
            timed_out=False,
            oom_killed=False,
        )


def _run_prlimit_only(
    command: list[str],
    *,
    config: SandboxConfig,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> SandboxResult:
    """Fallback: run with resource limits but no filesystem/network isolation."""
    if not hasattr(_run_prlimit_only, "_warned"):
        warnings.warn(
            "bwrap unavailable — running without OS-level sandboxing. "
            "Only prlimit resource limits will be applied.",
            RuntimeWarning,
            stacklevel=3,
        )
        _run_prlimit_only._warned = True

    full_env = dict(os.environ)
    if env:
        full_env.update(env)

    start = time.monotonic()
    timed_out = False

    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=full_env,
            preexec_fn=_make_preexec_fn(config),
        )
        try:
            stdout_bytes, stderr_bytes = proc.communicate(timeout=config.wall_time_limit_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_proc_tree(proc)
            try:
                stdout_bytes, stderr_bytes = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                stdout_bytes, stderr_bytes = b"", b""

        wall_time = time.monotonic() - start
        return SandboxResult(
            returncode=proc.returncode,
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            wall_time_s=wall_time,
            timed_out=timed_out,
            oom_killed=proc.returncode in (-9, 137),
        )
    except Exception as e:
        return SandboxResult(
            returncode=-1,
            stdout="",
            stderr=str(e),
            wall_time_s=time.monotonic() - start,
            timed_out=False,
            oom_killed=False,
        )


# ---------------------------------------------------------------------------
# Convenience: run a Python script in the sandbox
# ---------------------------------------------------------------------------


def run_python_sandboxed(
    script: str,
    *,
    config: SandboxConfig = _DEFAULT_CONFIG,
    script_dir: str | None = None,
    env: dict[str, str] | None = None,
) -> SandboxResult:
    """Write a Python script to a temp dir and execute it in the sandbox.

    If script_dir is provided, the script is written there and that dir
    must already be in config.writable_paths or config.extra_ro_binds.
    Otherwise a temp dir is created and added as a writable path.
    """
    paths = _discover_paths()
    python = paths["python"]

    # Prepend resource limits to the script
    full_script = _rlimit_preamble(config) + script

    if script_dir:
        script_path = os.path.join(script_dir, "_sandbox_runner.py")
        with open(script_path, "w") as f:
            f.write(full_script)
        return run_sandboxed(
            [python, script_path],
            config=config,
            cwd=script_dir,
            env=env,
        )

    # TemporaryDirectory guarantees cleanup even if finally itself raises
    with tempfile.TemporaryDirectory(prefix="cnake_sandbox_") as tmpdir:
        script_path = os.path.join(tmpdir, "_sandbox_runner.py")
        with open(script_path, "w") as f:
            f.write(full_script)

        # Add tmpdir as writable (for script + any output files)
        augmented = SandboxConfig(
            memory_limit_mb=config.memory_limit_mb,
            cpu_time_limit_s=config.cpu_time_limit_s,
            wall_time_limit_s=config.wall_time_limit_s,
            max_processes=config.max_processes,
            max_file_size_mb=config.max_file_size_mb,
            tmpfs_size_mb=config.tmpfs_size_mb,
            network=config.network,
            writable_paths=config.writable_paths + (tmpdir,),
            extra_ro_binds=config.extra_ro_binds,
            extra_env=config.extra_env,
        )

        return run_sandboxed(
            [python, script_path],
            config=augmented,
            cwd=tmpdir,
            env=env,
        )


# ---------------------------------------------------------------------------
# Convenience: run a sandbox_runners/*.py script with JSON config
# ---------------------------------------------------------------------------


def run_runner_sandboxed(
    runner_name: str,
    config_data: dict,
    *,
    config: SandboxConfig = _DEFAULT_CONFIG,
    env: dict[str, str] | None = None,
) -> SandboxResult:
    """Execute a runner from sandbox_runners/ with a JSON config file.

    Unlike run_python_sandboxed (which writes a script string to disk),
    this invokes an existing .py file from the sandbox_runners package
    and ro-binds it into the sandbox.  The runner is a real, lintable
    Python file — not an embedded string.

    Resource limits are injected into the config dict under ``_rlimits``
    so the runner can apply them via ``_common.apply_rlimits(config)``.

    Args:
        runner_name: Basename without extension, e.g. "correctness_runner".
        config_data: Dict to serialize as the runner's JSON config.
        config: SandboxConfig controlling resource limits and mounts.
        env: Extra environment variables for the sandbox.
    """
    from cnake_charmer.eval.sandbox_runners import RUNNERS_DIR

    runner_path = RUNNERS_DIR / f"{runner_name}.py"
    if not runner_path.exists():
        raise FileNotFoundError(f"Runner not found: {runner_path}")

    paths = _discover_paths()
    python = paths["python"]

    # Inject rlimits so the runner can apply them inside the sandbox
    config_data = {
        **config_data,
        "_rlimits": {
            "memory_mb": config.memory_limit_mb,
            "cpu_time_s": config.cpu_time_limit_s,
            "max_processes": config.max_processes,
            "max_file_size_mb": config.max_file_size_mb,
        },
    }

    with tempfile.TemporaryDirectory(prefix="cnake_runner_") as tmpdir:
        config_path = os.path.join(tmpdir, "_config.json")
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        augmented = SandboxConfig(
            memory_limit_mb=config.memory_limit_mb,
            cpu_time_limit_s=config.cpu_time_limit_s,
            wall_time_limit_s=config.wall_time_limit_s,
            max_processes=config.max_processes,
            max_file_size_mb=config.max_file_size_mb,
            tmpfs_size_mb=config.tmpfs_size_mb,
            network=config.network,
            writable_paths=config.writable_paths + (tmpdir,),
            extra_ro_binds=config.extra_ro_binds + (str(RUNNERS_DIR),),
            extra_env=config.extra_env,
        )

        return run_sandboxed(
            [python, str(runner_path), config_path],
            config=augmented,
            cwd=tmpdir,
            env=env,
        )


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------


def configure_logging():
    """Configure logging based on CNAKE_LOG_* environment variables.

    CNAKE_LOG_FORMAT: "json" for structured JSON lines, "text" for human-readable (default)
    CNAKE_LOG_LEVEL: DEBUG, INFO (default), WARNING, ERROR
    CNAKE_LOG_FILE: Optional file path for log output (with rotation)
    """
    log_format = os.environ.get("CNAKE_LOG_FORMAT", "text")
    log_level = os.environ.get("CNAKE_LOG_LEVEL", "INFO").upper()
    log_file = os.environ.get("CNAKE_LOG_FILE", "")

    root_logger = logging.getLogger("cnake_charmer")
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))

    if log_format == "json":
        formatter = logging.Formatter("%(message)s")
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler (if not already present)
    if not root_logger.handlers:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root_logger.addHandler(console)

    # File handler with rotation
    if log_file:
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
