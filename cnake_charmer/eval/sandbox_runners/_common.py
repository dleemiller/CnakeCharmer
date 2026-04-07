"""Shared helpers for sandbox runner scripts.

SANDBOX CONSTRAINTS — read before editing:

    1. These scripts execute inside a bwrap sandbox with no access to
       the host filesystem beyond explicitly mounted paths.
    2. They MUST NOT import from cnake_charmer — the repo root is not
       mounted.  Only stdlib, numpy, and venv packages are available.
    3. All helper code shared between runners lives in this file.
    4. Config is always a JSON file whose path is passed as sys.argv[1].
    5. All runner output MUST be a single JSON object printed to stdout.
       Anything on stderr is captured but only used for diagnostics.
    6. Each runner process exits after a single invocation — resource
       leaks (sys.path, open fds) are acceptable because the process
       is short-lived and sandboxed.
"""

import importlib
import importlib.util
import json
import os
import resource
import sys

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def load_config():
    """Read config JSON from sys.argv[1] and apply resource limits.

    Standard entry point for all runners.
    """
    config = json.loads(open(sys.argv[1]).read())  # noqa: SIM115
    apply_rlimits(config)
    return config


def load_both_funcs(config):
    """Load Python + Cython functions from config dict.

    Returns (py_func, cy_func).  On failure, prints error JSON and exits
    so the caller never sees a partial result.
    """
    try:
        py_func = load_python_func(config["python_code"], config["func_name"])
    except Exception as e:
        exit_error(f"Function {config['func_name']!r} not found in Python code: {e}")
    try:
        cy_func = load_cython_func(config["cython_module_path"], config["func_name"])
    except Exception as e:
        exit_error(f"Failed to load Cython module: {e}")
    return py_func, cy_func


def exit_error(msg):
    """Print an error JSON object and exit cleanly."""
    print(json.dumps({"error": msg}))
    sys.exit(0)


# ---------------------------------------------------------------------------
# Resource limits
# ---------------------------------------------------------------------------


def apply_rlimits(config):
    """Apply resource limits from the ``_rlimits`` key in the config dict.

    The key names here (``memory_mb``, ``cpu_time_s``, ``max_processes``,
    ``max_file_size_mb``) must match ``SandboxConfig.to_rlimits_dict()``
    in ``cnake_charmer/eval/sandbox.py``.  If you add a limit field,
    update both places.

    If the ``_rlimits`` key is missing the function is a no-op, so
    runners can also be tested outside the sandbox.
    """
    limits = config.get("_rlimits")
    if not limits:
        return
    mem_bytes = limits["memory_mb"] * 1024 * 1024
    fsize_bytes = limits["max_file_size_mb"] * 1024 * 1024
    if mem_bytes:  # 0 = unlimited (needed for ASan shadow memory)
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (limits["cpu_time_s"], limits["cpu_time_s"]))
    resource.setrlimit(resource.RLIMIT_NPROC, (limits["max_processes"], limits["max_processes"]))
    resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_bytes, fsize_bytes))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------


def load_module_from_path(path, name):
    """Dynamically load a Python/Cython module (.so) from *path*.

    Inserts the module's directory into sys.path so that transitive
    imports resolve.  This is fine because the process exits shortly
    after.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    mod_dir = os.path.dirname(os.path.abspath(path))
    if mod_dir not in sys.path:
        sys.path.insert(0, mod_dir)
    spec.loader.exec_module(mod)
    return mod


def load_python_func(python_code, func_name):
    """exec() *python_code* and return the callable named *func_name*."""
    namespace = {}
    exec(python_code, namespace)  # noqa: S102
    func = namespace.get(func_name)
    if func is None:
        raise LookupError(f"Function {func_name!r} not found in Python code")
    return func


def load_cython_func(cython_module_path, func_name):
    """Load a compiled Cython .so and return the callable named *func_name*."""
    so_basename = os.path.basename(cython_module_path)
    module_name = so_basename.split(".")[0]
    mod = load_module_from_path(cython_module_path, module_name)
    return getattr(mod, func_name)
