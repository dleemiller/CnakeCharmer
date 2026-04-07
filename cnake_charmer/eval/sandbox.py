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
from dataclasses import dataclass, field, replace
from functools import lru_cache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_BWRAP = shutil.which("bwrap")
_SANDBOX_ENABLED = os.environ.get("CNAKE_SANDBOX_ENABLED", "1" if _BWRAP else "0") == "1"


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
# Profiles
# ---------------------------------------------------------------------------

_PROFILES: dict[str, SandboxConfig] = {
    "compile": SandboxConfig(
        cpu_time_limit_s=120,
        wall_time_limit_s=150,
        memory_limit_mb=2048,
        max_file_size_mb=256,
        tmpfs_size_mb=512,
    ),
    "execute": SandboxConfig(
        cpu_time_limit_s=30,
        wall_time_limit_s=45,
        memory_limit_mb=2048,
        max_file_size_mb=64,
        tmpfs_size_mb=128,
    ),
    "asan": SandboxConfig(
        cpu_time_limit_s=60,
        wall_time_limit_s=90,
        memory_limit_mb=2048,
        max_file_size_mb=256,
        tmpfs_size_mb=512,
    ),
}


def compile_config(**overrides) -> SandboxConfig:
    """Profile for Cython compilation. Needs gcc, Python headers, Cython."""
    return replace(_PROFILES["compile"], **overrides)


def execute_config(**overrides) -> SandboxConfig:
    """Profile for running compiled code. Read-only — no writable bind mounts."""
    return replace(_PROFILES["execute"], **overrides)


def asan_config(**overrides) -> SandboxConfig:
    """Profile for AddressSanitizer. Needs ASan lib + compile toolchain."""
    from cnake_charmer.eval.memory_safety import _find_asan_lib

    asan_lib = _find_asan_lib()
    extra_ro = tuple(overrides.pop("extra_ro_binds", ()))
    if asan_lib:
        extra_ro = extra_ro + (asan_lib,)

    extra_env = dict(overrides.pop("extra_env", {}))
    extra_env.setdefault("PYTHONMALLOC", "malloc")
    extra_env.setdefault(
        "ASAN_OPTIONS",
        "detect_leaks=1:fast_unwind_on_malloc=0:print_legend=0:log_path=stderr",
    )
    if asan_lib:
        extra_env.setdefault("LD_PRELOAD", asan_lib)

    return replace(
        _PROFILES["asan"],
        extra_ro_binds=extra_ro,
        extra_env=extra_env,
        **overrides,
    )


# ---------------------------------------------------------------------------
# Path discovery (cached)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _discover_paths() -> dict[str, str]:
    """Discover and cache system paths needed for bwrap mounts."""
    return {"venv": sys.prefix, "python": sys.executable}


# ---------------------------------------------------------------------------
# Bwrap command construction (decomposed into readable sections)
# ---------------------------------------------------------------------------


def _namespace_flags(config: SandboxConfig) -> list[str]:
    """Isolation flags: namespaces, capabilities, session."""
    flags = [
        "--unshare-user",
        "--unshare-ipc",
        "--unshare-pid",
        "--unshare-uts",
        "--unshare-cgroup-try",
        "--disable-userns",
        "--die-with-parent",
        "--new-session",
    ]
    if not config.network:
        flags.append("--unshare-net")
    return flags


def _filesystem_mounts(config: SandboxConfig, venv: str) -> list[str]:
    """Filesystem layout: system libs, tmpfs, venv, caller-specified binds."""
    mounts = [
        # Core
        "--proc", "/proc",
        "--dev", "/dev",
        # System libraries (/lib, /lib64 are symlinks to /usr/*)
        "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64",
        # Linker cache
        "--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache",
        "--ro-bind-try", "/etc/alternatives", "/etc/alternatives",
        # Writable tmpfs
        "--size", str(config.tmpfs_size_mb * 1024 * 1024), "--tmpfs", "/tmp",
        "--size", str(10 * 1024 * 1024), "--tmpfs", "/home",
        # Venv (read-only)
        "--ro-bind", venv, venv,
    ]  # fmt: skip
    for path in config.extra_ro_binds:
        if path and os.path.exists(path):
            mounts += ["--ro-bind", path, path]
    for path in config.writable_paths:
        mounts += ["--bind", path, path]
    return mounts


def _environment_vars(config: SandboxConfig, venv: str, env: dict[str, str] | None) -> list[str]:
    """Environment: clearenv + whitelisted vars."""
    evars = [
        "--clearenv",
        "--setenv", "HOME", "/tmp",
        "--setenv", "PATH", f"{venv}/bin:/usr/bin:/usr/local/bin",
        "--setenv", "VIRTUAL_ENV", venv,
        "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
        "--setenv", "PYTHONPYCACHEPREFIX", "/tmp/__pycache__",
    ]  # fmt: skip
    for key, value in config.extra_env.items():
        evars += ["--setenv", key, value]
    if env:
        for key, value in env.items():
            evars += ["--setenv", key, value]
    return evars


def _build_bwrap_cmd(
    command: list[str],
    config: SandboxConfig,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> list[str]:
    """Build the full bwrap command line."""
    venv = _discover_paths()["venv"]
    cmd = [_BWRAP]
    cmd += _namespace_flags(config)
    cmd += _filesystem_mounts(config, venv)
    cmd += _environment_vars(config, venv, env)
    if cwd:
        cmd += ["--chdir", cwd]
    cmd += command
    return cmd


# ---------------------------------------------------------------------------
# Process lifecycle (shared by bwrap and fallback paths)
# ---------------------------------------------------------------------------


def _kill_proc_tree(proc: subprocess.Popen) -> None:
    """Kill a process and its entire process group. Best-effort, never raises."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except OSError:
        with contextlib.suppress(OSError):
            proc.kill()


def _execute(
    command: list[str],
    *,
    config: SandboxConfig,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    preexec_fn=None,
) -> SandboxResult:
    """Shared Popen lifecycle: spawn, communicate, timeout, kill, decode."""
    start = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
            preexec_fn=preexec_fn,
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

        return SandboxResult(
            returncode=proc.returncode,
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            wall_time_s=time.monotonic() - start,
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
# Public API
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = SandboxConfig()


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
        return _run_fallback(command, config=config, cwd=cwd, env=env)

    bwrap_cmd = _build_bwrap_cmd(command, config, cwd=cwd, env=env)
    cmd_summary = " ".join(command)[:200]
    _log_event("sandbox.start", command_summary=cmd_summary)

    # Apply inheritable rlimits (AS, CPU, FSIZE, CORE) to the bwrap
    # process — they propagate to the sandboxed child.  RLIMIT_NPROC is
    # deliberately excluded: clone() for namespace creation counts against
    # it, causing "Resource temporarily unavailable".  The PID namespace
    # provides equivalent fork-bomb protection.  Runner scripts also apply
    # all rlimits (including NPROC) from the _rlimits config key as a
    # belt-and-suspenders measure.
    result = _execute(bwrap_cmd, config=config, preexec_fn=_make_bwrap_preexec_fn(config))

    _log_event(
        "sandbox.complete",
        wall_time_s=round(result.wall_time_s, 3),
        returncode=result.returncode,
        timed_out=result.timed_out,
        oom_killed=result.oom_killed,
        command_summary=cmd_summary,
    )
    return result


def run_runner_sandboxed(
    runner_name: str,
    config_data: dict,
    *,
    config: SandboxConfig = _DEFAULT_CONFIG,
    env: dict[str, str] | None = None,
) -> SandboxResult:
    """Execute a runner from sandbox_runners/ with a JSON config file.

    The runner is a real, lintable Python file — ro-bound into the sandbox.
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

    python = _discover_paths()["python"]

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

        augmented = replace(
            config,
            writable_paths=config.writable_paths + (tmpdir,),
            extra_ro_binds=config.extra_ro_binds + (str(RUNNERS_DIR),),
        )

        return run_sandboxed(
            [python, str(runner_path), config_path],
            config=augmented,
            cwd=tmpdir,
            env=env,
        )


# ---------------------------------------------------------------------------
# Fallback (no bwrap)
# ---------------------------------------------------------------------------


def _make_preexec_fn(config: SandboxConfig):
    """Create a preexec_fn that sets process group + all resource limits."""
    mem_bytes = config.memory_limit_mb * 1024 * 1024
    fsize_bytes = config.max_file_size_mb * 1024 * 1024

    def _set_limits():
        os.setpgrp()
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (config.cpu_time_limit_s, config.cpu_time_limit_s))
        resource.setrlimit(resource.RLIMIT_NPROC, (config.max_processes, config.max_processes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_bytes, fsize_bytes))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return _set_limits


def _make_bwrap_preexec_fn(config: SandboxConfig):
    """Create a preexec_fn for the bwrap process itself.

    Sets process group + inheritable limits (AS, CPU, FSIZE, CORE).
    Excludes RLIMIT_NPROC because clone() for namespace creation counts
    against it.  The PID namespace provides equivalent fork-bomb protection.
    """
    mem_bytes = config.memory_limit_mb * 1024 * 1024
    fsize_bytes = config.max_file_size_mb * 1024 * 1024

    def _set_limits():
        os.setpgrp()
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (config.cpu_time_limit_s, config.cpu_time_limit_s))
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_bytes, fsize_bytes))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return _set_limits


def _run_fallback(
    command: list[str],
    *,
    config: SandboxConfig,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> SandboxResult:
    """Fallback: run with resource limits but no filesystem/network isolation."""
    if not hasattr(_run_fallback, "_warned"):
        warnings.warn(
            "bwrap unavailable — running without OS-level sandboxing. "
            "Only prlimit resource limits will be applied.",
            RuntimeWarning,
            stacklevel=3,
        )
        _run_fallback._warned = True

    full_env = dict(os.environ)
    if env:
        full_env.update(env)

    return _execute(
        command,
        config=config,
        cwd=cwd,
        env=full_env,
        preexec_fn=_make_preexec_fn(config),
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _log_event(event: str, profile: str = "", **extra):
    """Emit structured log event."""
    data = {"event": event, "profile": profile, **extra}
    logger.info(json.dumps(data, default=str))


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

    if not root_logger.handlers:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root_logger.addHandler(console)

    if log_file:
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
