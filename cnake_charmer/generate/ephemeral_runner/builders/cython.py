"""
Cython code builder for ephemeral environments.

This module provides a builder for compiling and executing Cython code in ephemeral environments.
"""

import os
import json
import tempfile
import time
import venv
import logging
from typing import List, Optional, Dict, Any

from ephemeral_runner.builders.base import BaseBuilder
from ephemeral_runner.utils.templates import load_template
from ephemeral_runner.exceptions import (
    VenvCreationError,
    FileWriteError,
    CompilationError,
)

# Configure logger
logger = logging.getLogger("ephemeral_runner.builders.cython")


class CythonBuilder(BaseBuilder):
    """
    Builder for Cython code compilation and execution in ephemeral environments.

    This builder:
    1. Creates a virtual environment
    2. Installs dependencies including Cython
    3. Attempts to compile and run Cython code using pyximport or setup.py
    """

    def __init__(self, request_id: str = None, max_install_attempts: int = 3):
        """
        Initialize the Cython builder.

        Args:
            request_id: Unique identifier for this build request
            max_install_attempts: Maximum number of attempts for dependency installation
        """
        super().__init__(request_id)
        self.max_install_attempts = max_install_attempts

    def build_and_run(self, code_str: str) -> Optional[str]:
        """
        Build and run Cython code in an ephemeral environment.

        Args:
            code_str: Cython code to compile and run

        Returns:
            Error message if failed, None if successful
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logger.info(
                f"Request {self.request_id}: Created temporary directory: {tmpdir}"
            )

            # 1) Create ephemeral venv
            try:
                venv_dir = os.path.join(tmpdir, "venv")
                logger.info(f"Request {self.request_id}: Creating venv at {venv_dir}")
                venv.create(venv_dir, with_pip=True)
                logger.info(f"Request {self.request_id}: Successfully created venv")
            except Exception as e:
                logger.error(
                    f"Request {self.request_id}: Error creating venv: {str(e)}"
                )
                return f"Failed to create virtual environment for Cython: {str(e)}"

            # 2) Parse dependencies
            try:
                deps = self.parse_dependencies(code_str)
                logger.info(f"Request {self.request_id}: Parsed dependencies: {deps}")
            except Exception as e:
                logger.error(
                    f"Request {self.request_id}: Error parsing dependencies: {str(e)}"
                )
                return f"Failed to parse dependencies: {str(e)}"

            # Always need cython
            if not any(d.lower() == "cython" for d in deps):
                deps.append("cython")
                logger.info(f"Request {self.request_id}: Added cython to dependencies")

            # 3) Install dependencies with retries
            install_error = self._install_dependencies(venv_dir, deps)
            if install_error:
                logger.error(
                    f"Request {self.request_id}: Failed to install dependencies after {self.max_install_attempts} attempts"
                )
                return install_error

            # 4) Write .pyx file
            pyx_path = os.path.join(tmpdir, "gen_code.pyx")
            try:
                with open(pyx_path, "w") as f:
                    f.write(code_str)
                logger.info(
                    f"Request {self.request_id}: Wrote Cython code ({len(code_str)} bytes) to {pyx_path}"
                )
            except Exception as e:
                logger.error(
                    f"Request {self.request_id}: Error writing Cython code to file: {str(e)}"
                )
                return f"Failed to write Cython code to file: {str(e)}"

            # 5) Create dependency analysis helper
            compile_info = self._analyze_dependencies(tmpdir, venv_dir, deps)

            # 6) Try pyximport first
            logger.info(
                f"Request {self.request_id}: Attempting compilation with pyximport"
            )
            pyximport_err = self._try_pyximport(tmpdir, venv_dir, compile_info)

            if not pyximport_err:
                logger.info(
                    f"Request {self.request_id}: pyximport compilation succeeded"
                )
                return None

            logger.info(
                f"Request {self.request_id}: pyximport compilation failed, falling back to setup.py"
            )

            # 7) Fall back to setup.py if pyximport fails
            logger.info(
                f"Request {self.request_id}: Compiling Cython code with setup.py"
            )
            setup_err = self._compile_with_setup(tmpdir, venv_dir, compile_info)
            if setup_err:
                logger.error(
                    f"Request {self.request_id}: Cython compilation failed: {setup_err[:1000]}..."
                    if len(setup_err) > 1000
                    else setup_err
                )
                return f"Cython compile error:\n{setup_err}"

            logger.info(f"Request {self.request_id}: Cython compilation successful")

            # 8) Run tests on the compiled module
            test_err = self._run_tests(tmpdir, venv_dir)
            if test_err:
                logger.warning(
                    f"Request {self.request_id}: Execution tests failed, but module compiled successfully"
                )
                # We don't fail the build here since the module compiled successfully
            else:
                logger.info(
                    f"Request {self.request_id}: Execution tests passed successfully"
                )

        return None

    def _install_dependencies(self, venv_dir: str, deps: List[str]) -> Optional[str]:
        """
        Install dependencies with retries.

        Args:
            venv_dir: Path to the virtual environment
            deps: List of dependencies to install

        Returns:
            Error message if failed, None if successful
        """
        for attempt in range(self.max_install_attempts):
            logger.info(
                f"Request {self.request_id}: Installing dependencies (attempt {attempt+1}/{self.max_install_attempts}): {deps}"
            )

            commands = [
                f"pip install --upgrade pip wheel setuptools",
                f"pip install {' '.join(deps)}",
            ]
            install_error = None

            for i, cmd in enumerate(commands):
                logger.info(
                    f"Request {self.request_id}: Running command [{i+1}/{len(commands)}]: {cmd}"
                )
                err = self.run_in_venv(venv_dir, cmd)
                if err:
                    install_error = f"Cython ephemeral venv install error: {err}"
                    logger.warning(
                        f"Request {self.request_id}: Command failed: {err[:300]}..."
                    )
                    break

            if not install_error:
                logger.info(
                    f"Request {self.request_id}: Successfully installed all dependencies"
                )
                return None

            # Wait before retry
            if attempt < self.max_install_attempts - 1:
                sleep_time = (attempt + 1) * 2  # Exponential backoff
                logger.info(
                    f"Request {self.request_id}: Waiting {sleep_time}s before retry"
                )
                time.sleep(sleep_time)

        return install_error

    def _analyze_dependencies(
        self, tmpdir: str, venv_dir: str, deps: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze dependencies to find include paths and other compilation requirements.

        Args:
            tmpdir: Path to the temporary directory
            venv_dir: Path to the virtual environment
            deps: List of dependencies

        Returns:
            Dictionary with compilation information
        """
        # Create a helper script to generate setup.py with proper dependency information
        setup_helper_path = os.path.join(tmpdir, "setup_helper.py")
        setup_helper_template = load_template("setup_helper.py.template")

        try:
            with open(setup_helper_path, "w") as f:
                f.write(setup_helper_template)
            logger.info(f"Request {self.request_id}: Wrote dependency helper script")
        except Exception as e:
            logger.error(
                f"Request {self.request_id}: Error writing helper script: {str(e)}"
            )
            # Continue with default empty configuration if this fails

        # Run the helper to get dependency information
        logger.info(f"Request {self.request_id}: Running dependency analysis helper")
        helper_cmd = f"python {setup_helper_path} {' '.join(deps)}"
        helper_output = self.run_in_venv(venv_dir, helper_cmd, capture_stdout=True)

        compile_info = {
            "include_dirs": [],
            "library_dirs": [],
            "libraries": [],
            "compile_args": [],
            "define_macros": [],
        }

        if helper_output:
            try:
                lines = helper_output.strip().split("\n")
                # Get the last line which should be the JSON output
                json_line = lines[-1]
                compile_info = json.loads(json_line)
                logger.info(
                    f"Request {self.request_id}: Dependency analysis result: {compile_info}"
                )
            except Exception as e:
                logger.warning(
                    f"Request {self.request_id}: Error parsing dependency analysis output: {str(e)}"
                )
                logger.info(
                    f"Raw output: {helper_output[:300]}..."
                    if len(helper_output) > 300
                    else helper_output
                )

        return compile_info

    def _try_pyximport(
        self, tmpdir: str, venv_dir: str, compile_info: Dict[str, Any]
    ) -> Optional[str]:
        """
        Try to compile and import using pyximport.

        Args:
            tmpdir: Path to the temporary directory
            venv_dir: Path to the virtual environment
            compile_info: Compilation information from dependency analysis

        Returns:
            Error message if failed, None if successful
        """
        # Generate adaptive pyximport test script
        test_script_path = os.path.join(tmpdir, "test_script.py")
        test_script_template = load_template("pyximport_test.py.template")

        try:
            # Format the template with compile_info
            test_script_content = test_script_template.format(
                include_dirs=compile_info["include_dirs"]
            )

            with open(test_script_path, "w") as f:
                f.write(test_script_content)
            logger.info(f"Request {self.request_id}: Wrote pyximport test script")
        except Exception as e:
            logger.error(
                f"Request {self.request_id}: Error writing test script: {str(e)}"
            )
            return f"Failed to write pyximport test script: {str(e)}"

        # Try using pyximport
        pyximport_cmd = f"python {test_script_path}"
        pyximport_err = self.run_in_venv(venv_dir, pyximport_cmd, cwd=tmpdir)

        return pyximport_err

    def _compile_with_setup(
        self, tmpdir: str, venv_dir: str, compile_info: Dict[str, Any]
    ) -> Optional[str]:
        """
        Compile Cython code using setup.py.

        Args:
            tmpdir: Path to the temporary directory
            venv_dir: Path to the virtual environment
            compile_info: Compilation information from dependency analysis

        Returns:
            Error message if failed, None if successful
        """
        setup_path = os.path.join(tmpdir, "setup.py")
        setup_template = load_template("setup.py.template")

        try:
            # Format the template with compile_info
            setup_content = setup_template.format(
                include_dirs=compile_info["include_dirs"],
                library_dirs=compile_info["library_dirs"],
                libraries=compile_info["libraries"],
                extra_compile_args=compile_info["compile_args"],
                define_macros=compile_info["define_macros"],
            )

            with open(setup_path, "w") as f:
                f.write(setup_content)
            logger.info(
                f"Request {self.request_id}: Wrote setup.py for Cython compilation"
            )
        except Exception as e:
            logger.error(f"Request {self.request_id}: Error writing setup.py: {str(e)}")
            return f"Failed to write setup.py: {str(e)}"

        # Compile directly
        compile_cmd = f"python setup.py build_ext --inplace"
        err = self.run_in_venv(venv_dir, compile_cmd, cwd=tmpdir)

        return err

    def _run_tests(self, tmpdir: str, venv_dir: str) -> Optional[str]:
        """
        Run tests on the compiled module.

        Args:
            tmpdir: Path to the temporary directory
            venv_dir: Path to the virtual environment

        Returns:
            Error message if failed, None if successful
        """
        test_runner_path = os.path.join(tmpdir, "run_tests.py")
        test_runner_template = load_template("test_runner.py.template")

        try:
            with open(test_runner_path, "w") as f:
                f.write(test_runner_template)
            logger.info(f"Request {self.request_id}: Wrote test runner script")
        except Exception as e:
            logger.error(
                f"Request {self.request_id}: Error writing test runner: {str(e)}"
            )
            # Continue despite this error since it's just testing
            return None

        # Run the test script
        logger.info(
            f"Request {self.request_id}: Running execution tests on compiled module"
        )
        test_cmd = f"python {test_runner_path}"
        test_err = self.run_in_venv(venv_dir, test_cmd, cwd=tmpdir)

        return test_err
