"""
Code Runner Module

This module provides a standalone system for building and running code in isolated environments.
It supports both Python and Cython code.
"""

import os
import re
import sys
import json
import time
import venv
import logging
import tempfile
import textwrap
import subprocess
import traceback
from typing import Dict, List, Set, Tuple, Optional, Union, Any

# Configure logger
logger = logging.getLogger("code_runner")

class CodeRunner:
    """
    A system for safely building and running code in isolated environments.
    
    This class handles:
    - Creating ephemeral virtual environments
    - Detecting and installing dependencies
    - Compiling and running Python or Cython code
    - Capturing output and errors
    """
    
    def __init__(self, 
                 max_install_attempts: int = 3, 
                 cleanup: bool = True,
                 timeout: int = 30):
        """
        Initialize the CodeRunner.
        
        Args:
            max_install_attempts: Maximum number of attempts for installing dependencies
            cleanup: Whether to clean up temporary files after execution
            timeout: Maximum time in seconds for code execution
        """
        self.max_install_attempts = max_install_attempts
        self.cleanup = cleanup
        self.timeout = timeout
        
        # Standard library modules that don't need to be installed
        self.stdlib_modules = {
            # Basic Python modules
            "sys", "os", "typing", "re", "subprocess", "traceback", "math", "time",
            # Collections and data structures
            "collections", "array", "dataclasses", "enum", "heapq", "queue", "bisect",
            # Threading and concurrency
            "threading", "multiprocessing", "concurrent", "asyncio", "_thread",
            # IO and file handling
            "io", "pathlib", "tempfile", "shutil", "fileinput", 
            # Data format handling
            "json", "csv", "pickle", "shelve", "sqlite3", "xml", "html",
            # Network and internet
            "socket", "ssl", "http", "urllib", "ftplib", "poplib", "imaplib", "smtplib", "email",
            # Date and time
            "datetime", "calendar", "zoneinfo",
            # Text processing
            "string", "textwrap", "difflib", "unicodedata",
            # Others
            "random", "itertools", "functools", "contextlib", "abc", "argparse", 
            "copy", "hashlib", "logging", "platform", "uuid", "weakref"
        }
        
        # System libraries that don't need to be installed
        self.system_libs = {"libc", "cpython", "libcpp", "posix"}
    
    def run_code(self, code_str: str) -> Dict[str, Any]:
        """
        Run the provided code string in an isolated environment.
        
        Args:
            code_str: The code string (Python or Cython) to run
            
        Returns:
            dict: Result containing stdout, stderr, and execution info
        """
        # Detect if it's Cython or Python
        is_cython = self._is_cython_code(code_str)
        
        logger.info(f"Running {'Cython' if is_cython else 'Python'} code ({len(code_str)} bytes)")
        
        # Create temporary directory for this run
        with tempfile.TemporaryDirectory() as tmpdir:
            if is_cython:
                result = self._build_and_run_cython(code_str, tmpdir)
            else:
                result = self._build_and_run_python(code_str, tmpdir)
            
            # Log detailed results
            if result["success"]:
                logger.info(f"Code execution successful (execution time: {result['execution_time']:.2f}s)")
            else:
                logger.error(f"Code execution failed: {result['stderr'][:500]}")
                
            return result
    
    def _build_and_run_python(self, code_str: str, tmpdir: str) -> Dict[str, Any]:
        """
        Build and run Python code in an isolated environment.
        
        Args:
            code_str: The Python code to run
            tmpdir: Temporary directory for this run
            
        Returns:
            dict: Result containing stdout, stderr, and execution info
        """
        result = {
            "language": "python",
            "success": False,
            "stdout": "",
            "stderr": "",
            "execution_time": 0,
            "dependencies": []
        }
        
        start_time = time.time()
        
        try:
            # 1) Create virtual environment
            venv_dir = os.path.join(tmpdir, "venv")
            try:
                venv.create(venv_dir, with_pip=True)
                logger.info(f"Created virtual environment at {venv_dir}")
            except Exception as e:
                error_msg = f"Failed to create virtual environment: {str(e)}"
                logger.error(error_msg)
                result["stderr"] = error_msg
                return result
            
            # 2) Upgrade pip and install setuptools
            logger.info("Upgrading pip and installing setuptools")
            bootstrap_cmd = "pip install --upgrade pip setuptools wheel"
            bootstrap_result = self._run_in_venv(venv_dir, bootstrap_cmd)
            
            if bootstrap_result.get("returncode", 1) != 0:
                error_msg = f"Failed to bootstrap environment: {bootstrap_result.get('stderr', '')}"
                logger.error(error_msg)
                result["stderr"] = error_msg
                return result
            logger.info("Environment bootstrapped successfully")
            
            # 3) Detect and install dependencies
            dependencies = self._parse_imports(code_str)
            result["dependencies"] = dependencies
            
            if dependencies:
                logger.info(f"Installing dependencies: {', '.join(dependencies)}")
                install_cmd = f"pip install {' '.join(dependencies)}"
                install_result = self._run_in_venv(venv_dir, install_cmd)
                
                if install_result.get("returncode", 1) != 0:
                    error_msg = f"Failed to install dependencies: {install_result.get('stderr', '')}"
                    logger.error(error_msg)
                    result["stderr"] = error_msg
                    return result
                logger.info("Dependencies installed successfully")
            
            # 4) Write code to file
            code_path = os.path.join(tmpdir, "code.py")
            with open(code_path, "w") as f:
                f.write(code_str)
            logger.info(f"Wrote code to {code_path}")
            
            # 5) Run the code
            logger.info(f"Running Python code (timeout: {self.timeout}s)")
            run_result = self._run_in_venv(venv_dir, f"python {code_path}", timeout=self.timeout)
            
            result["success"] = run_result.get("returncode", 1) == 0
            result["stdout"] = run_result.get("stdout", "")
            result["stderr"] = run_result.get("stderr", "")
            
            if result["success"]:
                logger.info("Python execution completed successfully")
            else:
                logger.error(f"Python execution failed: {result['stderr'][:200]}...")
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            result["stderr"] = error_msg
        
        result["execution_time"] = time.time() - start_time
        return result
    
    def _build_and_run_cython(self, code_str: str, tmpdir: str) -> Dict[str, Any]:
        """
        Build and run Cython code in an isolated environment.
        
        Args:
            code_str: The Cython code to run
            tmpdir: Temporary directory for this run
            
        Returns:
            dict: Result containing stdout, stderr, and execution info
        """
        result = {
            "language": "cython",
            "success": False,
            "stdout": "",
            "stderr": "",
            "execution_time": 0,
            "dependencies": [],
            "compilation_details": {}
        }
        
        start_time = time.time()
        
        try:
            # 1) Create virtual environment
            venv_dir = os.path.join(tmpdir, "venv")
            try:
                venv.create(venv_dir, with_pip=True)
                logger.info(f"Created virtual environment at {venv_dir}")
            except Exception as e:
                error_msg = f"Failed to create virtual environment: {str(e)}"
                logger.error(error_msg)
                result["stderr"] = error_msg
                return result
            
            # 2) Upgrade pip and install setuptools - CRITICAL for compilation!
            logger.info("Bootstrapping environment with pip, setuptools, and wheel")
            bootstrap_cmd = "pip install --upgrade pip setuptools wheel"
            bootstrap_result = self._run_in_venv(venv_dir, bootstrap_cmd)
            
            if bootstrap_result.get("returncode", 1) != 0:
                error_msg = f"Failed to bootstrap environment: {bootstrap_result.get('stderr', '')}"
                logger.error(error_msg)
                result["stderr"] = error_msg
                return result
            logger.info("Environment bootstrapped successfully")
            
            # 3) Detect and install dependencies
            dependencies = self._parse_imports(code_str)
            if "cython" not in dependencies:
                dependencies.append("cython")
            result["dependencies"] = dependencies
            
            # Install dependencies with retries
            install_success = False
            install_error = ""
            
            for attempt in range(self.max_install_attempts):
                logger.info(f"Installing dependencies (attempt {attempt+1}/{self.max_install_attempts}): {', '.join(dependencies)}")
                install_cmd = f"pip install {' '.join(dependencies)}"
                install_result = self._run_in_venv(venv_dir, install_cmd)
                
                if install_result.get("returncode", 1) == 0:
                    install_success = True
                    logger.info("Dependencies installed successfully")
                    break
                
                install_error = install_result.get("stderr", "Unknown error during installation")
                logger.warning(f"Dependency installation failed (attempt {attempt+1}): {install_error[:200]}...")
                time.sleep(2)  # Wait before retry
            
            if not install_success:
                error_msg = f"Failed to install dependencies after {self.max_install_attempts} attempts: {install_error}"
                logger.error(error_msg)
                result["stderr"] = error_msg
                result["compilation_details"]["dependency_error"] = install_error
                return result
            
            # 4) Write Cython code to file
            pyx_path = os.path.join(tmpdir, "code.pyx")
            with open(pyx_path, "w") as f:
                f.write(code_str)
            logger.info(f"Wrote Cython code to {pyx_path}")
            
            # 5) Create setup.py for compilation
            setup_path = os.path.join(tmpdir, "setup.py")
            with open(setup_path, "w") as f:
                f.write("""
from setuptools import setup, Extension
from Cython.Build import cythonize
import os
import sys

# Try to get NumPy include directory if NumPy is installed
numpy_include = []
try:
    import numpy
    numpy_include = [numpy.get_include()]
except ImportError:
    pass

extensions = [
    Extension(
        "code",
        ["code.pyx"],
        include_dirs=numpy_include
    )
]

setup(
    ext_modules=cythonize(extensions, language_level=3)
)
""")
            logger.info("Created setup.py for Cython compilation with NumPy support")
            
            # 6) Compile the Cython code
            logger.info("Compiling Cython code")
            compile_result = self._run_in_venv(venv_dir, "python setup.py build_ext --inplace", cwd=tmpdir)
            
            result["compilation_details"]["compile_stdout"] = compile_result.get("stdout", "")
            result["compilation_details"]["compile_stderr"] = compile_result.get("stderr", "")
            
            if compile_result.get("returncode", 1) != 0:
                error_msg = f"Failed to compile Cython code: {compile_result.get('stderr', '')}"
                logger.error(error_msg)
                result["stderr"] = error_msg
                return result
            
            logger.info("Cython compilation successful")
            
            # 7) Create a simple test script to run the compiled module
            test_script_path = os.path.join(tmpdir, "test_script.py")
            with open(test_script_path, "w") as f:
                f.write("""
import code
import inspect
import sys

# Get all callable functions from the module
functions = [name for name in dir(code) 
           if callable(getattr(code, name)) 
           and not name.startswith('_')]

if functions:
    print(f"Found callable functions: {', '.join(functions)}")
    
    # Try to call the first function with minimal arguments
    try:
        func = getattr(code, functions[0])
        sig = inspect.signature(func)
        args = {}
        
        # Create minimal arguments for each parameter
        for param_name, param in sig.parameters.items():
            # Simple defaults for common types
            if 'int' in str(param):
                args[param_name] = 0
            elif 'float' in str(param):
                args[param_name] = 0.0
            elif 'str' in str(param):
                args[param_name] = ""
            elif 'list' in str(param):
                args[param_name] = []
            elif 'dict' in str(param):
                args[param_name] = {}
            else:
                args[param_name] = None
        
        # Call the function
        result = func(**args)
        print(f"Called {functions[0]}() -> {result}")
    except Exception as e:
        print(f"Error calling {functions[0]}(): {e}")
        sys.exit(1)
else:
    print("No callable functions found in the module")
""")
            logger.info("Created test script for Cython module")
            
            # 8) Run the test script
            logger.info(f"Running test script (timeout: {self.timeout}s)")
            run_result = self._run_in_venv(venv_dir, f"python {test_script_path}", cwd=tmpdir, timeout=self.timeout)
            
            result["success"] = run_result.get("returncode", 1) == 0
            result["stdout"] = run_result.get("stdout", "")
            result["stderr"] = run_result.get("stderr", "")
            
            if result["success"]:
                logger.info("Cython module executed successfully")
            else:
                logger.error(f"Cython module execution failed: {result['stderr'][:200]}...")
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            result["stderr"] = error_msg
        
        result["execution_time"] = time.time() - start_time
        return result
    
    def _run_in_venv(self, 
                    venv_dir: str, 
                    command: str, 
                    cwd: Optional[str] = None,
                    timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a command in the virtual environment.
        
        Args:
            venv_dir: Path to the virtual environment
            command: Command to run
            cwd: Working directory for the command
            timeout: Timeout in seconds
            
        Returns:
            dict: Result containing returncode, stdout, and stderr
        """
        result = {
            "returncode": 1,
            "stdout": "",
            "stderr": ""
        }
        
        # Determine the Python executable path in the virtual environment
        if sys.platform == "win32":
            bin_dir = os.path.join(venv_dir, "Scripts")
        else:
            bin_dir = os.path.join(venv_dir, "bin")

        # Parse command to determine what executable to use
        if command.startswith("pip "):
            exe = os.path.join(bin_dir, "pip")
            args = command.split(" ", 1)[1].split()
        elif command.startswith("python "):
            exe = os.path.join(bin_dir, "python")
            args = command.split(" ", 1)[1].split()
        else:
            # Default to python for other commands
            exe = os.path.join(bin_dir, "python")
            args = command.split()

        cmd = [exe] + args
        cmd_str = " ".join(cmd)
        
        logger.debug(f"Running command in venv: {cmd_str}")
        
        try:
            process = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            result["returncode"] = process.returncode
            result["stdout"] = process.stdout
            result["stderr"] = process.stderr
            
            if process.returncode == 0:
                logger.debug(f"Command succeeded: {cmd_str}")
                if process.stdout:
                    logger.debug(f"Command stdout: {process.stdout[:300]}..." 
                              if len(process.stdout) > 300 else process.stdout)
            else:
                logger.debug(f"Command failed with code {process.returncode}: {cmd_str}")
                if process.stderr:
                    logger.debug(f"Command stderr: {process.stderr[:300]}..." 
                              if len(process.stderr) > 300 else process.stderr)
            
        except subprocess.TimeoutExpired:
            error_msg = f"Command timed out after {timeout} seconds"
            logger.error(error_msg)
            result["stderr"] = error_msg
        except Exception as e:
            error_msg = f"Failed to run command: {str(e)}"
            logger.error(error_msg)
            result["stderr"] = error_msg
        
        return result
    
    def _parse_imports(self, code_str: str) -> List[str]:
        """
        Parse import statements from code to determine dependencies.
        
        Args:
            code_str: The code string to parse
            
        Returns:
            list: List of required pip packages
        """
        libs = set()
        
        try:
            # Pattern for regular imports and cimports
            import_pattern = re.compile(
                r"^(?:cimport|import|from)\s+([a-zA-Z0-9_\.]+)",
                re.MULTILINE
            )
            matches = import_pattern.findall(code_str)
            
            for m in matches:
                top_level = m.split(".")[0]
                if top_level not in self.stdlib_modules and top_level not in self.system_libs:
                    libs.add(top_level)
            
            # Common library aliases and their actual package names
            common_aliases = {
                "np": "numpy",
                "pd": "pandas",
                "plt": "matplotlib",
                "tf": "tensorflow",
                "torch": "torch",
                "sk": "scikit-learn",
                "sp": "scipy"
            }
            
            # Check for common library aliases
            if self._is_cython_code(code_str):
                for alias, lib_name in common_aliases.items():
                    if f"{alias}." in code_str and lib_name not in libs:
                        libs.add(lib_name)
            
            dependencies = sorted(libs)
            logger.info(f"Detected dependencies: {', '.join(dependencies) if dependencies else 'none'}")
            return dependencies
        except Exception as e:
            logger.error(f"Error parsing dependencies: {str(e)}")
            return ["cython"] if self._is_cython_code(code_str) else []
    
    def _is_cython_code(self, code_str: str) -> bool:
        """
        Check if code appears to be Cython.
        
        Args:
            code_str: The code string to check
            
        Returns:
            bool: True if the code contains Cython-specific elements
        """
        cython_indicators = ["cdef", "cpdef", "cimport", "nogil", "# cython:"]
        return any(indicator in code_str for indicator in cython_indicators)


def run_code(code_str: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Convenience function to run code without creating a CodeRunner instance.
    
    Args:
        code_str: The code string to run
        timeout: Maximum execution time in seconds
        
    Returns:
        dict: Result containing stdout, stderr, and execution info
    """
    runner = CodeRunner(timeout=timeout)
    return runner.run_code(code_str)


def run_cython_with_annotation(code_str: str) -> Dict[str, Any]:
    """
    Run Cython code and generate HTML annotation.
    
    Args:
        code_str: The Cython code to run
        
    Returns:
        dict: Result containing execution info and annotation HTML
    """
    # Standard run result
    result = run_code(code_str)
    
    # Only try to generate annotation if it's Cython code
    if "cython" in code_str.lower() or any(kw in code_str for kw in ["cdef", "cpdef", "cimport"]):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write code to file
            pyx_path = os.path.join(tmpdir, "code.pyx")
            with open(pyx_path, "w") as f:
                f.write(code_str)
            
            # Create setup.py with annotation enabled
            setup_path = os.path.join(tmpdir, "setup.py")
            with open(setup_path, "w") as f:
                f.write("""
from setuptools import setup, Extension
from Cython.Build import cythonize
import os
import sys

# Try to get NumPy include directory if NumPy is installed
numpy_include = []
try:
    import numpy
    numpy_include = [numpy.get_include()]
except ImportError:
    pass

extensions = [
    Extension(
        "code",
        ["code.pyx"],
        include_dirs=numpy_include
    )
]

setup(
    ext_modules=cythonize(extensions, language_level=3, annotate=True)
)
""")
            
            # Create virtual environment
            venv_dir = os.path.join(tmpdir, "venv")
            venv.create(venv_dir, with_pip=True)
            
            # Upgrade pip and install setuptools
            runner = CodeRunner()
            bootstrap_result = runner._run_in_venv(venv_dir, "pip install --upgrade pip setuptools wheel")
            
            if bootstrap_result["returncode"] == 0:
                # Install Cython and NumPy if needed
                if "numpy" in code_str:
                    install_result = runner._run_in_venv(venv_dir, "pip install cython numpy")
                else:
                    install_result = runner._run_in_venv(venv_dir, "pip install cython")
                
                if install_result["returncode"] == 0:
                    # Build with annotation
                    build_result = runner._run_in_venv(venv_dir, "python setup.py build_ext", cwd=tmpdir)
                    
                    # Check if annotation was generated
                    html_path = pyx_path + ".html"
                    if os.path.exists(html_path):
                        with open(html_path, "r") as f:
                            result["annotation_html"] = f.read()
                    else:
                        result["annotation_error"] = "No annotation HTML was generated"
                else:
                    result["annotation_error"] = "Failed to install Cython for annotation"
            else:
                result["annotation_error"] = "Failed to bootstrap environment for annotation"
    
    return result 