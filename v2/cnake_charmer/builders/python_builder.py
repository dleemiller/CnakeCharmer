# builders/python_builder.py
import os
import sys
import tempfile
import time
import importlib.util
import traceback
from typing import Dict, List, Optional, Any

from builders.base import BaseBuilder
from core.models import BuildResult, ExecutionResult
from core.enums import LanguageType


class PythonBuilder(BaseBuilder):
    """Builder for Python code."""
    
    def supported_languages(self) -> List[LanguageType]:
        """Get supported languages."""
        return [LanguageType.PYTHON]
    
    def build(self, code: str, language: LanguageType, options: Dict) -> BuildResult:
        """
        Build Python code (compile to bytecode).
        
        Args:
            code: Python code
            language: Must be PYTHON
            options: Build options
            
        Returns:
            Build result
        """
        if language != LanguageType.PYTHON:
            return BuildResult(
                success=False,
                output="",
                error=f"Unsupported language: {language}",
                build_time=0.0
            )
        
        start_time = time.time()
        
        # Create a temporary file
        fd, path = tempfile.mkstemp(suffix='.py')
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(code)
            
            # Try to compile the code
            try:
                with open(path, 'r') as f:
                    source = f.read()
                
                compile(source, path, 'exec')
                
                build_time = time.time() - start_time
                
                return BuildResult(
                    success=True,
                    output="Python code compiled successfully",
                    artifact_path=path,
                    build_time=build_time
                )
            
            except SyntaxError as e:
                error_msg = f"Syntax error: {str(e)}"
                build_time = time.time() - start_time
                
                return BuildResult(
                    success=False,
                    output="",
                    error=error_msg,
                    build_time=build_time
                )
            
            except Exception as e:
                error_msg = f"Compilation error: {str(e)}"
                build_time = time.time() - start_time
                
                return BuildResult(
                    success=False,
                    output="",
                    error=error_msg,
                    build_time=build_time
                )
        
        except Exception as e:
            build_time = time.time() - start_time
            
            return BuildResult(
                success=False,
                output="",
                error=f"Error creating temporary file: {str(e)}",
                build_time=build_time
            )
    
    def run(self, build_artifact: str, inputs: Dict) -> ExecutionResult:
        """
        Run Python code.
        
        Args:
            build_artifact: Path to the Python file
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
                # Load the module
                spec = importlib.util.spec_from_file_location("dynamic_module", build_artifact)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Look for a main function
                if hasattr(module, 'main'):
                    result = module.main(**inputs)
                else:
                    # If no main function, the output is just what was printed
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