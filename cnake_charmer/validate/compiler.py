"""
Compile Cython code in an ephemeral temp directory.

Returns structured results with success/failure, error messages, and
optionally the path to the compiled module and HTML annotation.
"""

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

logger = logging.getLogger(__name__)

SETUP_PY_TEMPLATE = """\
from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        [Extension("{module_name}", ["{module_name}.pyx"])],
        compiler_directives={{
            "language_level": "3",
            "boundscheck": {boundscheck},
            "wraparound": {wraparound},
        }},
        annotate={annotate},
    ),
)
"""


@dataclass
class CompilationResult:
    success: bool
    errors: str = ""
    warnings: str = ""
    build_dir: str | None = None
    module_path: str | None = None
    html_path: str | None = None


def compile_cython(
    code: str,
    module_name: str = "gen_module",
    annotate: bool = True,
    boundscheck: bool = False,
    wraparound: bool = False,
    keep_build: bool = False,
    extra_deps: list | None = None,
    timeout: int = 120,
) -> CompilationResult:
    """
    Compile a Cython code string in an ephemeral temp directory.

    Args:
        code: The .pyx source code as a string.
        module_name: Name for the generated module.
        annotate: Generate HTML annotation file.
        boundscheck: Enable bounds checking.
        wraparound: Enable negative index wraparound.
        keep_build: If True, don't delete the temp directory (for debugging).
        extra_deps: Additional pip packages to install before compiling.
        timeout: Subprocess timeout in seconds.

    Returns:
        CompilationResult with success status, errors, and paths.
    """
    tmpdir = tempfile.mkdtemp(prefix="cnake_compile_")
    result = CompilationResult(success=False, build_dir=tmpdir)

    try:
        # Write the .pyx file
        pyx_path = os.path.join(tmpdir, f"{module_name}.pyx")
        with open(pyx_path, "w") as f:
            f.write(code)

        # Write setup.py
        setup_content = SETUP_PY_TEMPLATE.format(
            module_name=module_name,
            boundscheck=boundscheck,
            wraparound=wraparound,
            annotate=annotate,
        )
        setup_path = os.path.join(tmpdir, "setup.py")
        with open(setup_path, "w") as f:
            f.write(setup_content)

        # Install extra dependencies if needed
        if extra_deps:
            dep_cmd = _get_install_cmd(extra_deps)
            dep_result = subprocess.run(
                dep_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if dep_result.returncode != 0:
                result.errors = f"Failed to install dependencies: {dep_result.stderr}"
                return result

        # Compile using the same Python that's running this process
        compile_result = subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            capture_output=True,
            text=True,
            cwd=tmpdir,
            timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

        # Capture warnings from stderr even on success
        result.warnings = compile_result.stderr

        if compile_result.returncode != 0:
            result.errors = compile_result.stderr or compile_result.stdout
            logger.debug(f"Compilation failed: {result.errors[:500]}")
            return result

        # Find the compiled .so/.pyd file
        for f in os.listdir(tmpdir):
            if f.startswith(module_name) and (f.endswith(".so") or f.endswith(".pyd")):
                result.module_path = os.path.join(tmpdir, f)
                break

        # Find HTML annotation
        html_path = os.path.join(tmpdir, f"{module_name}.html")
        if os.path.exists(html_path):
            result.html_path = html_path

        if result.module_path:
            result.success = True
            logger.debug(f"Compilation succeeded: {result.module_path}")
        else:
            result.errors = "Compilation appeared to succeed but no .so/.pyd file found"

        return result

    except subprocess.TimeoutExpired:
        result.errors = f"Compilation timed out after {timeout}s"
        return result
    except Exception as e:
        result.errors = str(e)
        return result
    finally:
        if not keep_build and not result.success:
            shutil.rmtree(tmpdir, ignore_errors=True)
            result.build_dir = None


def _get_install_cmd(packages: list) -> list:
    """Get the install command, preferring uv if available."""
    uv = shutil.which("uv")
    if uv:
        return [uv, "pip", "install", "--quiet"] + packages
    return [sys.executable, "-m", "pip", "install", "--quiet"] + packages


def cleanup_build(result: CompilationResult) -> None:
    """Clean up the build directory from a CompilationResult."""
    if result.build_dir and os.path.exists(result.build_dir):
        shutil.rmtree(result.build_dir, ignore_errors=True)
        result.build_dir = None
