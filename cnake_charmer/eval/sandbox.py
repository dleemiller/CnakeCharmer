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

from cnake_charmer.eval.bwrap import build_command as _build_bwrap_cmd

logger = logging.getLogger(__name__)
_SANDBOX_EVENT_LOGS_ENABLED = os.environ.get("CNAKE_SANDBOX_LOG_EVENTS", "0") == "1"

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

    def to_rlimits_dict(self) -> dict[str, int]:
        """Serialize resource limits for injection into runner config JSON.

        The keys here are the contract with sandbox_runners/_common.py's
        apply_rlimits().  If you add a limit field, update both this
        method and apply_rlimits().
        """
        return {
            "memory_mb": self.memory_limit_mb,
            "cpu_time_s": self.cpu_time_limit_s,
            "max_processes": self.max_processes,
            "max_file_size_mb": self.max_file_size_mb,
        }


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
        memory_limit_mb=0,  # ASan needs ~15TB virtual address space for shadow memory
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
# Bwrap command construction lives in bwrap.py (pure functions, no I/O)
# ---------------------------------------------------------------------------


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
    """Shared Popen lifecycle: spawn, communicate, timeout, kill, decode.

    State transitions::

        SPAWN → RUNNING → COMPLETED
                        → TIMED_OUT → KILL_TREE → DRAIN(5s) → DONE
                                                 → FORCE_KILL(5s) → DONE
        SPAWN → ERROR
    """
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
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=5)
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

    bwrap_cmd = _build_bwrap_cmd(
        command, config, bwrap_path=_BWRAP, venv=_discover_paths()["venv"], cwd=cwd, env=env
    )
    cmd_summary = " ".join(command)[:200]
    _log_event("sandbox.start", command_summary=cmd_summary)

    # Apply inheritable rlimits (AS, CPU, FSIZE, CORE) to the bwrap
    # process — they propagate to the sandboxed child.  RLIMIT_NPROC is
    # deliberately excluded: clone() for namespace creation counts against
    # it, causing "Resource temporarily unavailable".  The PID namespace
    # provides equivalent fork-bomb protection.  Runner scripts also apply
    # all rlimits (including NPROC) from the _rlimits config key as a
    # belt-and-suspenders measure.
    result = _execute(
        bwrap_cmd, config=config, preexec_fn=_make_preexec_fn(config, exclude_nproc=True)
    )

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
        "_rlimits": config.to_rlimits_dict(),
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


def _make_preexec_fn(config: SandboxConfig, *, exclude_nproc: bool = False):
    """Create a preexec_fn that sets process group + resource limits.

    Args:
        config: Sandbox configuration with limit values.
        exclude_nproc: If True, skip RLIMIT_NPROC.  Required for bwrap
            because clone() for namespace creation counts against it.
    """
    mem_bytes = config.memory_limit_mb * 1024 * 1024
    fsize_bytes = config.max_file_size_mb * 1024 * 1024

    def _set_limits():
        os.setpgrp()
        if mem_bytes:  # 0 = unlimited (needed for ASan shadow memory)
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (config.cpu_time_limit_s, config.cpu_time_limit_s))
        if not exclude_nproc:
            resource.setrlimit(resource.RLIMIT_NPROC, (config.max_processes, config.max_processes))
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
    if not _SANDBOX_EVENT_LOGS_ENABLED:
        return
    data = {"event": event, "profile": profile, **extra}
    logger.info(json.dumps(data, default=str))
