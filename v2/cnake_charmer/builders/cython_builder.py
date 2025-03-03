# builders/cython_builder.py
import os
import sys
import tempfile
import time
import subprocess
import importlib.util
import traceback
from typing import Dict, List, Optional, Any

from builders.base import BaseBuilder
from core.models import BuildResult, ExecutionResult
from core.enums import LanguageType


class CythonBuilder(BaseBuilder):
    """Builder for Cython code."""
    
    def supported_languages(self) -> List[LanguageType]:
        """Get supported languages."""
        return [LanguageType.CYTHON]
    
    def build(self, code: str, language: LanguageType, options: Dict) -> BuildResult:
        """
        Build Cython code.
        
        Args:
            code: Cython code
            language: Must be CYTHON
            options: Build options
            
        Returns:
            Build result
        """
        if language != LanguageType.CYTHON:
            return BuildResult(
                success=False,
                output="",
                error=f"Unsupported language: {language}",
                build_time=0.0
            )
        
        start_time = time.time()
        
        # Create a temporary directory for the build
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create the Cython file
            cython_file = os.path.join(temp_dir, "module.pyx")
            with open(cython_file, 'w') as f:
                f.write(code)
            
            # Create a setup.py file
            setup_file = os.path.join(temp_dir, "setup.py")
            with open(setup_file, 'w') as f:
                f.write("""
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("module.pyx"),
    zip_safe=False,
)
""")
            
            # Run the build
            cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
            process = subprocess.Popen(
                cmd,
                cwd=temp_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                build_time = time.time() - start_time
                
                return BuildResult(
                    success=False,
                    output=stdout,
                    error=stderr,
                    build_time=build_time
                )
            
            # Find the built module
            import glob
            module_files = glob.glob(os.path.join(temp_dir, "module*.so"))
            
            if not module_files:
                build_time = time.time() - start_time
                
                return BuildResult(
                    success=False,
                    output=stdout,
                    error="Built module not found",
                    build_time=build_time
                )
            
            build_time = time.time() - start_time
            
            return BuildResult(
                success=True,
                output=stdout,
                artifact_path=module_files[0],
                build_time=build_time
            )
        
        except Exception as e:
            build_time = time.time() - start_time
            
            return BuildResult(
                success=False,
                output="",
                error=f"Error building Cython code: {str(e)}",
                build_time=build_time
            )
    
    def run(self, build_artifact: str, inputs: Dict) -> ExecutionResult:
        """
        Run Cython code.
        
        Args:
            build_artifact: Path to the compiled Cython module
            inputs: Execution inputs
            
        Returns:
            Execution result
        """
        start_time = time.time()
        
        try:
            # Redirect stdout and stderr
            import io
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            new_stdout = io.StringIO()
            new_stderr = io.StringIO()
            sys.stdout = new_stdout
            sys.stderr = new_stderr
            
            try:
                # Extract the module name from the artifact path
                module_name = os.path.basename(build_artifact).split('.')[0]
                
                # Add the directory to sys.path
                module_dir = os.path.dirname(build_artifact)
                if module_dir not in sys.path:
                    sys.path.insert(0, module_dir)
                
                # Import the module
                module = __import__(module_name)
                
                # Look for a main function
                if hasattr(module, 'main'):
                    result = module.main(**inputs)
                else:
                    # Try to call a function with the same name as one of the inputs
                    for func_name in inputs.keys():
                        if hasattr(module, func_name):
                            func = getattr(module, func_name)
                            if callable(func):
                                result = func(inputs[func_name])
                                break
                    else:
                        # If no matching function, just get all callable attributes
                        callables = {
                            name: getattr(module, name)
                            for name in dir(module)
                            if callable(getattr(module, name)) and not name.startswith('_')
                        }
                        if callables:
                            func_name, func = next(iter(callables.items()))
                            # Try to call with the first input value
                            if inputs:
                                first_input = next(iter(inputs.values()))
                                result = func(first_input)
                            else:
                                result = func()
                        else:
                            result = None
                
                execution_time = time.time() - start_time
                
                return ExecutionResult(
                    success=True,
                    output=new_stdout.getvalue(),
                    execution_time=execution_time
                )
            
            except Exception as e:
                error_msg = f"Execution error: {str(e)}\n{traceback.format_exc()}"
                execution_time = time.time() - start_time
                
                return ExecutionResult(
                    success=False,
                    output=new_stdout.getvalue(),
                    error=error_msg,
                    execution_time=execution_time
                )
            
            finally:
                # Restore stdout and stderr
                sys.stdout = old_stdout
                sys.stderr = old_stderr
        
        except Exception as e:
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=False,
                output="",
                error=f"Error setting up execution environment: {str(e)}",
                execution_time=execution_time
            )