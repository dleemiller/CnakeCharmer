# builders/container.py
import os
import subprocess
import tempfile
import logging
import time
from typing import Dict, List, Optional, Any

from core.models import BuildResult, ExecutionResult
from core.enums import LanguageType
from core.exceptions import BuildError


class ContainerBuilder:
    """Builder that uses containers for isolation."""
    
    def __init__(self, image: str = "python:3.9"):
        """
        Initialize container builder.
        
        Args:
            image: Docker image to use
        """
        self.image = image
        self.logger = logging.getLogger(__name__)
    
    def supported_languages(self) -> List[LanguageType]:
        """Get supported languages."""
        return [LanguageType.PYTHON, LanguageType.CYTHON]
    
    def build(self, code: str, language: LanguageType, options: Dict) -> BuildResult:
        """
        Build code in a container.
        
        Args:
            code: Code to build
            language: Language of the code
            options: Build options
            
        Returns:
            Build result
        """
        if language not in self.supported_languages():
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
            # Create the code file
            if language == LanguageType.PYTHON:
                code_file = os.path.join(temp_dir, "module.py")
            elif language == LanguageType.CYTHON:
                code_file = os.path.join(temp_dir, "module.pyx")
            else:
                code_file = os.path.join(temp_dir, "module.txt")
            
            with open(code_file, 'w') as f:
                f.write(code)
            
            # Create a setup.py file for Cython
            if language == LanguageType.CYTHON:
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
            
            # Create a Dockerfile
            dockerfile = os.path.join(temp_dir, "Dockerfile")
            with open(dockerfile, 'w') as f:
                f.write(f"""
FROM {self.image}

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir pytest pytest-cov
""")
                
                if language == LanguageType.CYTHON:
                    f.write("RUN pip install --no-cache-dir cython\n")
                    f.write("RUN python setup.py build_ext --inplace\n")
                elif language == LanguageType.PYTHON:
                    f.write("RUN python -m py_compile module.py\n")
            
            # Build the container
            container_name = f"build_{int(time.time())}"
            cmd = ["docker", "build", "-t", container_name, temp_dir]
            
            self.logger.info(f"Building container: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
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
            
            build_time = time.time() - start_time
            
            return BuildResult(
                success=True,
                output=stdout,
                artifact_path=container_name,
                build_time=build_time
            )
        
        except Exception as e:
            build_time = time.time() - start_time
            
            return BuildResult(
                success=False,
                output="",
                error=f"Error building code in container: {str(e)}",
                build_time=build_time
            )
    
    def run(self, build_artifact: str, inputs: Dict) -> ExecutionResult:
        """
        Run code in a container.
        
        Args:
            build_artifact: Container name
            inputs: Execution inputs
            
        Returns:
            Execution result
        """
        start_time = time.time()
        
        try:
            # Create a temporary file for inputs
            fd, inputs_file = tempfile.mkstemp(suffix='.json')
            os.close(fd)
            
            with open(inputs_file, 'w') as f:
                import json
                json.dump(inputs, f)
            
            # Run the container
            cmd = [
                "docker", "run", "--rm",
                "-v", f"{inputs_file}:/app/inputs.json",
                build_artifact,
                "python", "-c", "import json; import module; with open('/app/inputs.json', 'r') as f: inputs = json.load(f); result = module.main(**inputs) if hasattr(module, 'main') else None; print(result)"
            ]
            
            self.logger.info(f"Running container: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate()
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=process.returncode == 0,
                output=stdout,
                error=stderr if process.returncode != 0 else None,
                execution_time=execution_time
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=False,
                output="",
                error=f"Error running code in container: {str(e)}",
                execution_time=execution_time
            )
        
        finally:
            # Clean up inputs file
            if 'inputs_file' in locals() and os.path.exists(inputs_file):
                os.unlink(inputs_file)
    
    def cleanup(self, artifact_path: str) -> None:
        """
        Clean up container.
        
        Args:
            artifact_path: Container name
        """
        try:
            subprocess.run(["docker", "rmi", artifact_path], check=False)
            self.logger.info(f"Removed container image: {artifact_path}")
        except Exception as e:
            self.logger.warning(f"Error removing container image {artifact_path}: {e}")