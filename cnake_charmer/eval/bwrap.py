"""Bubblewrap (bwrap) command-line construction.

Builds the argument list for bwrap from a SandboxConfig.  Pure functions —
no I/O, no subprocess calls, no global state.  Testable without bwrap
installed.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cnake_charmer.eval.sandbox import SandboxConfig


def namespace_flags(config: SandboxConfig) -> list[str]:
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


def filesystem_mounts(config: SandboxConfig, venv: str) -> list[str]:
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


def environment_vars(config: SandboxConfig, venv: str, env: dict[str, str] | None) -> list[str]:
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


def build_command(
    command: list[str],
    config: SandboxConfig,
    *,
    bwrap_path: str,
    venv: str,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> list[str]:
    """Build the full bwrap command line.

    Args:
        command: The command to run inside the sandbox.
        config: SandboxConfig controlling mounts and environment.
        bwrap_path: Absolute path to the bwrap binary.
        venv: Absolute path to the Python venv to mount read-only.
        cwd: Working directory inside the sandbox.
        env: Extra environment variables (merged after config.extra_env).
    """
    cmd = [bwrap_path]
    cmd += namespace_flags(config)
    cmd += filesystem_mounts(config, venv)
    cmd += environment_vars(config, venv, env)
    if cwd:
        cmd += ["--chdir", cwd]
    cmd += command
    return cmd
