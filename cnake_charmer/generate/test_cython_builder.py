import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock, mock_open
import json

from ephemeral_runner.builders.cython import CythonBuilder
from ephemeral_runner.exceptions import (
    VenvCreationError,
    FileWriteError,
    CompilationError,
)


class TestCythonBuilder(unittest.TestCase):
    """Test cases for the CythonBuilder class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.builder = CythonBuilder(request_id="test-123")
        self.sample_code = """
# cython: language_level=3
import numpy as np

def hello_world():
    return "Hello World from Cython!"

def add_arrays(a, b):
    cdef int i, n
    n = len(a)
    result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        result[i] = a[i] + b[i]
    return result
"""
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after each test method."""
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("venv.create")
    def test_init(self, mock_venv_create):
        """Test the constructor."""
        builder = CythonBuilder(request_id="custom-id", max_install_attempts=5)
        self.assertEqual(builder.request_id, "custom-id")
        self.assertEqual(builder.max_install_attempts, 5)

    @patch("ephemeral_runner.builders.cython.tempfile.TemporaryDirectory")
    @patch("ephemeral_runner.builders.cython.venv.create")
    def test_build_and_run_venv_creation_error(self, mock_venv_create, mock_temp_dir):
        """Test handling of venv creation errors."""
        mock_temp_dir.return_value.__enter__.return_value = self.tmpdir
        mock_venv_create.side_effect = Exception("Venv creation failed")

        result = self.builder.build_and_run(self.sample_code)

        mock_venv_create.assert_called_once()
        self.assertIn("Failed to create virtual environment", result)
        self.assertIn("Venv creation failed", result)

    @patch("ephemeral_runner.builders.cython.tempfile.TemporaryDirectory")
    @patch("ephemeral_runner.builders.cython.venv.create")
    def test_build_and_run_dependency_parse_error(
        self, mock_venv_create, mock_temp_dir
    ):
        """Test handling of dependency parsing errors."""
        mock_temp_dir.return_value.__enter__.return_value = self.tmpdir

        # Mock the parse_dependencies method to raise an exception
        with patch.object(
            self.builder,
            "parse_dependencies",
            side_effect=Exception("Parse deps failed"),
        ):
            result = self.builder.build_and_run(self.sample_code)

        mock_venv_create.assert_called_once()
        self.assertIn("Failed to parse dependencies", result)
        self.assertIn("Parse deps failed", result)

    @patch("ephemeral_runner.builders.cython.tempfile.TemporaryDirectory")
    @patch("ephemeral_runner.builders.cython.venv.create")
    @patch("builtins.open", new_callable=mock_open)
    def test_build_and_run_write_error(
        self, mock_file, mock_venv_create, mock_temp_dir
    ):
        """Test handling of file write errors."""
        mock_temp_dir.return_value.__enter__.return_value = self.tmpdir

        # Mock parse_dependencies
        with patch.object(
            self.builder, "parse_dependencies", return_value=["numpy", "cython"]
        ):
            # Mock _install_dependencies
            with patch.object(self.builder, "_install_dependencies", return_value=None):
                # Force file write to fail
                mock_file.side_effect = Exception("File write failed")

                result = self.builder.build_and_run(self.sample_code)

        self.assertIn("Failed to write Cython code to file", result)
        self.assertIn("File write failed", result)

    @patch("ephemeral_runner.builders.cython.tempfile.TemporaryDirectory")
    @patch("ephemeral_runner.builders.cython.venv.create")
    def test_build_and_run_dependencies_installation_error(
        self, mock_venv_create, mock_temp_dir
    ):
        """Test handling of dependency installation errors."""
        mock_temp_dir.return_value.__enter__.return_value = self.tmpdir

        # Mock parse_dependencies
        with patch.object(
            self.builder, "parse_dependencies", return_value=["numpy", "cython"]
        ):
            # Mock _install_dependencies to return an error
            with patch.object(
                self.builder,
                "_install_dependencies",
                return_value="Failed to install dependencies",
            ):

                result = self.builder.build_and_run(self.sample_code)

        self.assertEqual(result, "Failed to install dependencies")

    @patch("ephemeral_runner.builders.cython.tempfile.TemporaryDirectory")
    @patch("ephemeral_runner.builders.cython.venv.create")
    @patch("builtins.open", new_callable=mock_open)
    def test_build_and_run_pyximport_success(
        self, mock_file, mock_venv_create, mock_temp_dir
    ):
        """Test successful compilation with pyximport."""
        mock_temp_dir.return_value.__enter__.return_value = self.tmpdir

        # Setup mocks
        with patch.object(
            self.builder, "parse_dependencies", return_value=["numpy", "cython"]
        ):
            with patch.object(self.builder, "_install_dependencies", return_value=None):
                with patch.object(
                    self.builder,
                    "_analyze_dependencies",
                    return_value={
                        "include_dirs": [],
                        "library_dirs": [],
                        "libraries": [],
                        "compile_args": [],
                        "define_macros": [],
                    },
                ):
                    with patch.object(
                        self.builder, "_try_pyximport", return_value=None
                    ):
                        # Mock _run_tests
                        with patch.object(
                            self.builder, "_run_tests", return_value=None
                        ):
                            result = self.builder.build_and_run(self.sample_code)

        self.assertIsNone(result)

    @patch("ephemeral_runner.builders.cython.tempfile.TemporaryDirectory")
    @patch("ephemeral_runner.builders.cython.venv.create")
    @patch("builtins.open", new_callable=mock_open)
    def test_build_and_run_setup_fallback_success(
        self, mock_file, mock_venv_create, mock_temp_dir
    ):
        """Test fallback to setup.py when pyximport fails."""
        mock_temp_dir.return_value.__enter__.return_value = self.tmpdir

        # Setup mocks
        with patch.object(
            self.builder, "parse_dependencies", return_value=["numpy", "cython"]
        ):
            with patch.object(self.builder, "_install_dependencies", return_value=None):
                with patch.object(
                    self.builder,
                    "_analyze_dependencies",
                    return_value={
                        "include_dirs": [],
                        "library_dirs": [],
                        "libraries": [],
                        "compile_args": [],
                        "define_macros": [],
                    },
                ):
                    with patch.object(
                        self.builder, "_try_pyximport", return_value="pyximport failed"
                    ):
                        with patch.object(
                            self.builder, "_compile_with_setup", return_value=None
                        ):
                            with patch.object(
                                self.builder, "_run_tests", return_value=None
                            ):
                                result = self.builder.build_and_run(self.sample_code)

        self.assertIsNone(result)

    @patch("ephemeral_runner.builders.cython.tempfile.TemporaryDirectory")
    @patch("ephemeral_runner.builders.cython.venv.create")
    @patch("builtins.open", new_callable=mock_open)
    def test_build_and_run_setup_failure(
        self, mock_file, mock_venv_create, mock_temp_dir
    ):
        """Test handling of setup.py compilation failure."""
        mock_temp_dir.return_value.__enter__.return_value = self.tmpdir

        # Setup mocks
        with patch.object(
            self.builder, "parse_dependencies", return_value=["numpy", "cython"]
        ):
            with patch.object(self.builder, "_install_dependencies", return_value=None):
                with patch.object(
                    self.builder,
                    "_analyze_dependencies",
                    return_value={
                        "include_dirs": [],
                        "library_dirs": [],
                        "libraries": [],
                        "compile_args": [],
                        "define_macros": [],
                    },
                ):
                    with patch.object(
                        self.builder, "_try_pyximport", return_value="pyximport failed"
                    ):
                        with patch.object(
                            self.builder,
                            "_compile_with_setup",
                            return_value="setup.py compilation failed",
                        ):
                            result = self.builder.build_and_run(self.sample_code)

        self.assertIn("Cython compile error", result)
        self.assertIn("setup.py compilation failed", result)

    @patch("ephemeral_runner.builders.cython.tempfile.TemporaryDirectory")
    @patch("ephemeral_runner.builders.cython.venv.create")
    @patch("builtins.open", new_callable=mock_open)
    def test_build_and_run_test_failure_not_fatal(
        self, mock_file, mock_venv_create, mock_temp_dir
    ):
        """Test that test failures don't cause build failures."""
        mock_temp_dir.return_value.__enter__.return_value = self.tmpdir

        # Setup mocks
        with patch.object(
            self.builder, "parse_dependencies", return_value=["numpy", "cython"]
        ):
            with patch.object(self.builder, "_install_dependencies", return_value=None):
                with patch.object(
                    self.builder,
                    "_analyze_dependencies",
                    return_value={
                        "include_dirs": [],
                        "library_dirs": [],
                        "libraries": [],
                        "compile_args": [],
                        "define_macros": [],
                    },
                ):
                    with patch.object(
                        self.builder, "_try_pyximport", return_value=None
                    ):
                        with patch.object(
                            self.builder,
                            "_run_tests",
                            return_value="Test failed but that's ok",
                        ):
                            result = self.builder.build_and_run(self.sample_code)

        # Build should still succeed even though tests failed
        self.assertIsNone(result)

    def test_parse_dependencies(self):
        """Test dependency parsing from code string."""
        code_with_deps = """
# cython: language_level=3
# deps: numpy,pandas,matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def process_data(data):
    return np.mean(data)
"""
        deps = self.builder.parse_dependencies(code_with_deps)

        # Should extract dependencies from the deps comment
        # Use set comparison since order doesn't matter
        self.assertSetEqual(set(deps), set(["numpy", "pandas", "matplotlib"]))

        # Test with no deps comment
        code_no_deps = "def hello(): return 'world'"
        deps = self.builder.parse_dependencies(code_no_deps)
        self.assertListEqual(deps, [])

    @patch("ephemeral_runner.builders.cython.time.sleep")
    def test_install_dependencies_success(self, mock_sleep):
        """Test successful dependency installation."""
        venv_dir = "/fake/venv/dir"
        deps = ["numpy", "cython"]

        # Mock run_in_venv to succeed
        with patch.object(self.builder, "run_in_venv", return_value=None) as mock_run:
            result = self.builder._install_dependencies(venv_dir, deps)

            # Should call run_in_venv twice (pip upgrade and pip install)
            self.assertEqual(mock_run.call_count, 2)
            self.assertIsNone(result)
            # Sleep should not be called on success
            mock_sleep.assert_not_called()

    @patch("ephemeral_runner.builders.cython.time.sleep")
    def test_install_dependencies_retry(self, mock_sleep):
        """Test dependency installation with retries."""
        venv_dir = "/fake/venv/dir"
        deps = ["numpy", "cython"]

        # First attempt fails, second succeeds
        side_effects = [
            "Error on first command",  # First attempt, first command
            None,  # Second attempt, first command
            None,  # Second attempt, second command
        ]

        with patch.object(
            self.builder, "run_in_venv", side_effect=side_effects
        ) as mock_run:
            result = self.builder._install_dependencies(venv_dir, deps)

            # Should call run_in_venv three times (failure, then two successes)
            self.assertEqual(mock_run.call_count, 3)
            self.assertIsNone(result)
            # Sleep should be called once for retry
            mock_sleep.assert_called_once_with(2)  # First retry with 2 seconds

    @patch("ephemeral_runner.builders.cython.time.sleep")
    def test_install_dependencies_max_retries(self, mock_sleep):
        """Test dependency installation with max retries reached."""
        venv_dir = "/fake/venv/dir"
        deps = ["numpy", "cython"]
        builder = CythonBuilder(request_id="test-retry", max_install_attempts=3)

        # All attempts fail
        with patch.object(
            builder, "run_in_venv", return_value="Installation error"
        ) as mock_run:
            result = builder._install_dependencies(venv_dir, deps)

            # Should call run_in_venv three times (once per attempt)
            self.assertEqual(mock_run.call_count, 3)
            self.assertIn("Cython ephemeral venv install error", result)
            # Sleep should be called twice for retries
            self.assertEqual(mock_sleep.call_count, 2)
            # Verify exponential backoff
            mock_sleep.assert_any_call(2)  # First retry
            mock_sleep.assert_any_call(4)  # Second retry

    @patch("ephemeral_runner.builders.base.BaseBuilder.run_in_venv")
    @patch("ephemeral_runner.builders.cython.load_template")
    @patch("builtins.open", new_callable=mock_open)
    def test_analyze_dependencies(
        self, mock_file, mock_load_template, mock_run_in_venv
    ):
        """Test dependency analysis helper function."""
        tmpdir = "/fake/tmp/dir"
        venv_dir = "/fake/venv/dir"
        deps = ["numpy", "cython"]

        # Mock template
        mock_load_template.return_value = "# Template content"

        # Mock run_in_venv to return JSON
        compile_info = {
            "include_dirs": ["/usr/include/python3.8"],
            "library_dirs": ["/usr/lib"],
            "libraries": ["python3.8"],
            "compile_args": ["-O3"],
            "define_macros": [["NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"]],
        }
        mock_run_in_venv.return_value = f"Some output\n{json.dumps(compile_info)}"

        result = self.builder._analyze_dependencies(tmpdir, venv_dir, deps)

        mock_file.assert_called_once()
        mock_run_in_venv.assert_called_once()
        self.assertEqual(result, compile_info)

    @patch("ephemeral_runner.builders.base.BaseBuilder.run_in_venv")
    @patch("ephemeral_runner.builders.cython.load_template")
    @patch("builtins.open", new_callable=mock_open)
    def test_analyze_dependencies_error(
        self, mock_file, mock_load_template, mock_run_in_venv
    ):
        """Test dependency analysis with JSON parsing error."""
        tmpdir = "/fake/tmp/dir"
        venv_dir = "/fake/venv/dir"
        deps = ["numpy", "cython"]

        # Mock template
        mock_load_template.return_value = "# Template content"

        # Mock run_in_venv to return invalid JSON
        mock_run_in_venv.return_value = "Some output\nNot valid JSON"

        result = self.builder._analyze_dependencies(tmpdir, venv_dir, deps)

        # Should return default empty configuration
        self.assertEqual(result["include_dirs"], [])
        self.assertEqual(result["library_dirs"], [])
        self.assertEqual(result["libraries"], [])
        self.assertEqual(result["compile_args"], [])
        self.assertEqual(result["define_macros"], [])

    @patch("ephemeral_runner.builders.base.BaseBuilder.run_in_venv")
    @patch("ephemeral_runner.builders.cython.load_template")
    @patch("builtins.open", new_callable=mock_open)
    def test_try_pyximport_success(
        self, mock_file, mock_load_template, mock_run_in_venv
    ):
        """Test successful pyximport compilation."""
        tmpdir = "/fake/tmp/dir"
        venv_dir = "/fake/venv/dir"
        compile_info = {
            "include_dirs": ["/usr/include/python3.8"],
            "library_dirs": [],
            "libraries": [],
            "compile_args": [],
            "define_macros": [],
        }

        # Mock template
        mock_load_template.return_value = "# Template with {include_dirs}"

        # Mock run_in_venv to succeed
        mock_run_in_venv.return_value = None

        result = self.builder._try_pyximport(tmpdir, venv_dir, compile_info)

        mock_file.assert_called_once()
        mock_run_in_venv.assert_called_once()
        self.assertIsNone(result)

    @patch("ephemeral_runner.builders.base.BaseBuilder.run_in_venv")
    @patch("ephemeral_runner.builders.cython.load_template")
    @patch("builtins.open", new_callable=mock_open)
    def test_try_pyximport_failure(
        self, mock_file, mock_load_template, mock_run_in_venv
    ):
        """Test pyximport compilation failure."""
        tmpdir = "/fake/tmp/dir"
        venv_dir = "/fake/venv/dir"
        compile_info = {
            "include_dirs": ["/usr/include/python3.8"],
            "library_dirs": [],
            "libraries": [],
            "compile_args": [],
            "define_macros": [],
        }

        # Mock template
        mock_load_template.return_value = "# Template with {include_dirs}"

        # Mock run_in_venv to fail
        mock_run_in_venv.return_value = "Pyximport error: compilation failed"

        result = self.builder._try_pyximport(tmpdir, venv_dir, compile_info)

        mock_file.assert_called_once()
        mock_run_in_venv.assert_called_once()
        self.assertEqual(result, "Pyximport error: compilation failed")

    @patch("ephemeral_runner.builders.base.BaseBuilder.run_in_venv")
    @patch("ephemeral_runner.builders.cython.load_template")
    @patch("builtins.open", new_callable=mock_open)
    def test_compile_with_setup_success(
        self, mock_file, mock_load_template, mock_run_in_venv
    ):
        """Test successful setup.py compilation."""
        tmpdir = "/fake/tmp/dir"
        venv_dir = "/fake/venv/dir"
        compile_info = {
            "include_dirs": ["/usr/include/python3.8"],
            "library_dirs": ["/usr/lib"],
            "libraries": ["python3.8"],
            "compile_args": ["-O3"],
            "define_macros": [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        }

        # Mock template
        mock_load_template.return_value = "# Template with {include_dirs}, {library_dirs}, {libraries}, {extra_compile_args}, {define_macros}"

        # Mock run_in_venv to succeed
        mock_run_in_venv.return_value = None

        result = self.builder._compile_with_setup(tmpdir, venv_dir, compile_info)

        mock_file.assert_called_once()
        mock_run_in_venv.assert_called_once()
        self.assertIsNone(result)

    @patch("ephemeral_runner.builders.base.BaseBuilder.run_in_venv")
    @patch("ephemeral_runner.builders.cython.load_template")
    @patch("builtins.open", new_callable=mock_open)
    def test_compile_with_setup_failure(
        self, mock_file, mock_load_template, mock_run_in_venv
    ):
        """Test setup.py compilation failure."""
        tmpdir = "/fake/tmp/dir"
        venv_dir = "/fake/venv/dir"
        compile_info = {
            "include_dirs": ["/usr/include/python3.8"],
            "library_dirs": ["/usr/lib"],
            "libraries": ["python3.8"],
            "compile_args": ["-O3"],
            "define_macros": [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        }

        # Mock template
        mock_load_template.return_value = "# Template with {include_dirs}, {library_dirs}, {libraries}, {extra_compile_args}, {define_macros}"

        # Mock run_in_venv to fail
        mock_run_in_venv.return_value = "Setup.py error: compilation failed"

        result = self.builder._compile_with_setup(tmpdir, venv_dir, compile_info)

        mock_file.assert_called_once()
        mock_run_in_venv.assert_called_once()
        self.assertEqual(result, "Setup.py error: compilation failed")

    @patch("ephemeral_runner.builders.base.BaseBuilder.run_in_venv")
    @patch("ephemeral_runner.builders.cython.load_template")
    @patch("builtins.open", new_callable=mock_open)
    def test_run_tests_success(self, mock_file, mock_load_template, mock_run_in_venv):
        """Test successful test execution."""
        tmpdir = "/fake/tmp/dir"
        venv_dir = "/fake/venv/dir"

        # Mock template
        mock_load_template.return_value = "# Test runner template"

        # Mock run_in_venv to succeed
        mock_run_in_venv.return_value = None

        result = self.builder._run_tests(tmpdir, venv_dir)

        mock_file.assert_called_once()
        mock_run_in_venv.assert_called_once()
        self.assertIsNone(result)

    @patch("ephemeral_runner.builders.base.BaseBuilder.run_in_venv")
    @patch("ephemeral_runner.builders.cython.load_template")
    @patch("builtins.open", new_callable=mock_open)
    def test_run_tests_failure(self, mock_file, mock_load_template, mock_run_in_venv):
        """Test test execution failure."""
        tmpdir = "/fake/tmp/dir"
        venv_dir = "/fake/venv/dir"

        # Mock template
        mock_load_template.return_value = "# Test runner template"

        # Mock run_in_venv to fail
        mock_run_in_venv.return_value = "Test error: module import failed"

        result = self.builder._run_tests(tmpdir, venv_dir)

        mock_file.assert_called_once()
        mock_run_in_venv.assert_called_once()
        self.assertEqual(result, "Test error: module import failed")

    @patch("ephemeral_runner.builders.base.BaseBuilder.run_in_venv")
    @patch("ephemeral_runner.builders.cython.load_template")
    def test_run_tests_file_write_error(self, mock_load_template, mock_run_in_venv):
        """Test handling of file write errors during test execution."""
        tmpdir = "/fake/tmp/dir"
        venv_dir = "/fake/venv/dir"

        # Mock template
        mock_load_template.return_value = "# Test runner template"

        # Mock file open to fail
        with patch("builtins.open", side_effect=Exception("File write failed")):
            result = self.builder._run_tests(tmpdir, venv_dir)

        # Should continue despite the error
        self.assertIsNone(result)
        # run_in_venv should not be called if file write fails
        mock_run_in_venv.assert_not_called()


if __name__ == "__main__":
    unittest.main()
