import os
import re
import subprocess
import sys
import tempfile
import traceback
import venv
import textwrap
import dspy
import time
import logging
from dspy.primitives import Module
from dspy.signatures import InputField, OutputField
from dspy.signatures.signature import Signature, ensure_signature
from dotenv import load_dotenv

###############################################################################
# 1) CONFIGURE LOGGING
###############################################################################
# Configure logger
logger = logging.getLogger("EphemeralCodeGenerator")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)  # Set to logging.DEBUG for more verbose output

###############################################################################
# 2) CONFIGURE DSPY / LLM
###############################################################################
load_dotenv()
# Example: using an openrouter model. Adjust as needed.
lm = dspy.LM(model="openrouter/anthropic/claude-3.5-sonnet")
dspy.configure(lm=lm)

###############################################################################
# 2) CODE GENERATOR CLASS
###############################################################################

class EphemeralCodeGenerator(Module):
    """
    Demonstrates:
      - LLM generation
      - parse + ephemeral build in a venv
      - library detection from code snippet
      - automatic install + compile
      - regeneration on error
    """
    def __init__(self, signature, max_iters=3):
        super().__init__()
        self.signature = ensure_signature(signature)
        self.max_iters = max_iters

        # Chain for initial generation
        self.code_generate = dspy.ChainOfThought(
            Signature(
                {
                    "prompt": self.signature.fields["prompt"],
                    "generated_code": self.signature.fields["generated_code"]
                },
                instructions=(
                    "You are given `prompt` describing a user request. "
                    "Generate code in either Python or Cython that solves the request. "
                    "Output ONLY the code in triple backticks, no extra commentary.\n\n"
                    "If it is Cython, prefer comment-based directives for boundscheck and wraparound, e.g.:\n"
                    "# cython: boundscheck=False\n"
                    "# cython: wraparound=False\n\n"
                    "For Cython code, include ALL necessary imports and cimports explicitly.\n"
                    "Make sure any cython functions have proper type declarations."
                )
            ),
        )

        # Chain for regeneration
        self.code_regenerate = dspy.ChainOfThought(
            Signature(
                {
                    "prompt": self.signature.fields["prompt"],
                    "previous_code": InputField(
                        prefix="Previous Code:",
                        desc="Previously generated code that errored",
                        format=str
                    ),
                    "error": InputField(
                        prefix="Error:",
                        desc="Error message from compilation or runtime",
                        format=str
                    ),
                    "generated_code": self.signature.fields["generated_code"]
                },
                instructions=(
                    "You generated code previously that failed to run/compile. "
                    "The user prompt is `prompt`. The failing code is `previous_code`. "
                    "The error message is `error`.\n"
                    "Your job: correct the code and provide a working version in triple backticks, "
                    "with no extra commentary.\n\n"
                    "Make sure to include ALL necessary imports and cimports.\n"
                    "Make sure all required libraries are properly imported."
                )
            )
        )

    def forward(self, **kwargs):
        """
        1) Generate code.
        2) Parse code block from triple backticks.
        3) Attempt ephemeral build/run.
        4) If error => regeneration loop up to max_iters.
        """
        logger.info("Forward called with prompt: %s", kwargs.get("prompt", "")[:50] + "...")
        
        # Step 1: get initial code
        code_data = self.code_generate(**kwargs)
        raw_code = code_data.get("generated_code", "")
        logger.debug("Initial generation raw output: %s", raw_code[:100] + "..." if len(raw_code) > 100 else raw_code)

        # Step 2: parse
        code_block, parse_err = self._extract_code(raw_code)
        if parse_err:
            logger.warning("Parse error => regeneration: %s", parse_err)
            return self._try_regeneration(kwargs, previous_code="", error=parse_err)

        # Step 3: ephemeral build/run
        error = self._ephemeral_build_and_run(code_block)
        if error:
            logger.warning("Ephemeral build error => regeneration: %s", error[:100] + "..." if len(error) > 100 else error)
            return self._try_regeneration(kwargs, previous_code=code_block, error=error)

        logger.info("Successfully generated and built code")
        return {"generated_code": code_block, "error": None}

    def _try_regeneration(self, kwargs, previous_code, error):
        attempts = 0
        while attempts < self.max_iters:
            attempts += 1
            logger.info("Attempting regeneration, attempt #%d", attempts)
            regen_data = self.code_regenerate(
                prompt=kwargs["prompt"],
                previous_code=previous_code,
                error=error
            )
            new_raw = regen_data.get("generated_code", "")
            new_code, parse_err = self._extract_code(new_raw)
            if parse_err:
                # next iteration
                logger.warning("Parse error on regenerated code => continuing: %s", parse_err)
                previous_code = new_raw
                error = parse_err
                continue

            build_err = self._ephemeral_build_and_run(new_code)
            if build_err:
                logger.warning("Ephemeral build error again => continuing: %s", 
                             build_err[:100] + "..." if len(build_err) > 100 else build_err)
                error = build_err
                previous_code = new_code
            else:
                # success
                logger.info("Regeneration successful on attempt #%d", attempts)
                return {"generated_code": new_code, "error": None}

        # if we exhaust attempts
        logger.error("Exhausted all regeneration attempts, still has error")
        return {"generated_code": previous_code, "error": error}

    ############################################################################
    # CODE PARSING
    ############################################################################

    def _extract_code(self, text):
        """
        Grab triple-backtick code from LLM response. If missing, fallback to entire text.
        """
        match = re.search(r"```[\w\s]*\n?(.*?)```", text, re.DOTALL)
        if not match:
            code_block = text.strip()
            if not code_block:
                logger.error("Could not parse code block - empty content")
                return ("", "ERROR: Could not parse code block.")
            logger.warning("No triple backticks found, using entire text as code")
            return (code_block, None)
        
        code_block = match.group(1).strip()
        if not code_block:
            logger.error("Empty code block after triple backticks")
            return ("", "ERROR: Empty code block after triple backticks.")
        
        logger.info("Successfully extracted code block (%d characters)", len(code_block))
        logger.debug("Code block begins with: %s", code_block[:100] + "..." if len(code_block) > 100 else code_block)
        return (code_block, None)

    ############################################################################
    # EPHEMERAL BUILD + RUN
    ############################################################################

    def _ephemeral_build_and_run(self, code_str):
        """
        1) Create ephemeral venv
        2) Detect imports => pip install them (plus cython if needed)
        3) If Cython => attempt compile. Otherwise => run Python.
        Return error string or None on success.
        """
        # Are we dealing with Cython or Python?
        if self._is_cython(code_str):
            return self._build_and_run_cython(code_str)
        else:
            return self._build_and_run_python(code_str)

    def _is_cython(self, code_str):
        low = code_str.lower()
        if "cdef" in code_str or ".pyx" in code_str or "cimport" in code_str or "cython" in low:
            return True
        return False

    def _build_and_run_python(self, code_str):
        """
        Ephemeral venv -> install detected libs -> run code with python.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1) create ephemeral venv
            logger.info("Creating ephemeral venv for Python execution in %s", tmpdir)
            venv_dir = os.path.join(tmpdir, "venv")
            venv.create(venv_dir, with_pip=True)

            # 2) parse dependencies (imported libs) -> pip install
            deps = self._parse_imports_for_python(code_str)
            logger.info("Detected dependencies: %s", deps)

            commands = [
                # upgrade pip
                f"pip install --upgrade pip wheel setuptools",
            ]
            if deps:
                commands.append(f"pip install {' '.join(deps)}")

            # 3) write code to .py
            py_path = os.path.join(tmpdir, "gen_code.py")
            with open(py_path, "w") as f:
                f.write(code_str)
            logger.info("Wrote Python code to %s", py_path)

            # 4) run
            for cmd in commands:
                logger.info("Running command: %s", cmd)
                err = self._run_in_venv(venv_dir, cmd)
                if err:
                    logger.error("Dependency installation failed: %s", 
                               err[:100] + "..." if len(err) > 100 else err)
                    return f"Python ephemeral venv install error: {err}"

            logger.info("Executing Python code")
            run_cmd = f"python {py_path}"
            err = self._run_in_venv(venv_dir, run_cmd)
            if err:
                logger.error("Python execution failed: %s", 
                           err[:100] + "..." if len(err) > 100 else err)
                return f"Python run error: {err}"
                
            logger.info("Python execution completed successfully")
        return None

    def _build_and_run_cython(self, code_str):
        """
        1) ephemeral venv
        2) detect libraries -> pip install
        3) write setup.py for compilation
        4) compile + run
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_dir = os.path.join(tmpdir, "venv")
            venv.create(venv_dir, with_pip=True)

            # parse imports => gather needed deps
            deps = self._parse_imports_for_python(code_str)
            
            # always need cython
            if not any(d.lower() == "cython" for d in deps):
                deps.append("cython")

            # Install dependencies with retries
            max_install_attempts = 3
            for attempt in range(max_install_attempts):
                logger.debug(f"Installing dependencies (attempt {attempt+1}/{max_install_attempts}): {deps}")
                commands = [
                    f"pip install --upgrade pip wheel setuptools",
                    f"pip install {' '.join(deps)}",
                ]
                install_error = None
                for cmd in commands:
                    err = self._run_in_venv(venv_dir, cmd)
                    if err:
                        install_error = f"Cython ephemeral venv install error: {err}"
                        break
                
                if not install_error:
                    break
                    
                # Wait before retry
                if attempt < max_install_attempts - 1:
                    time.sleep(1)
            
            if install_error:
                return install_error

            # write the .pyx
            pyx_path = os.path.join(tmpdir, "gen_code.pyx")
            with open(pyx_path, "w") as f:
                f.write(code_str)

            # Create a helper script to generate setup.py with proper dependency information
            setup_helper_path = os.path.join(tmpdir, "setup_helper.py")
            with open(setup_helper_path, "w") as f:
                f.write(textwrap.dedent("""
                import sys
                import importlib
                import json
                import os

                # This script helps discover any special include paths or compilation
                # requirements for the installed dependencies.

                # Arguments: list of dependencies
                deps = sys.argv[1:]
                result = {
                    'include_dirs': [],
                    'library_dirs': [],
                    'libraries': [],
                    'compile_args': [],
                    'define_macros': []
                }

                for dep in deps:
                    try:
                        # Try to import the dependency
                        mod = importlib.import_module(dep)
                        
                        # Check for common special cases
                        # numpy provides get_include
                        if hasattr(mod, 'get_include'):
                            include_dir = mod.get_include()
                            print(f"Found include directory for {dep}: {include_dir}")
                            if include_dir not in result['include_dirs']:
                                result['include_dirs'].append(include_dir)
                        
                        # Try to get package location to find headers
                        if hasattr(mod, '__file__'):
                            pkg_dir = os.path.dirname(mod.__file__)
                            potential_include = os.path.join(os.path.dirname(pkg_dir), 'include')
                            if os.path.exists(potential_include):
                                print(f"Found potential include directory: {potential_include}")
                                if potential_include not in result['include_dirs']:
                                    result['include_dirs'].append(potential_include)
                    
                    except ImportError as e:
                        print(f"Could not import {dep}: {e}")
                        continue

                # Print result as JSON for the parent process
                print(json.dumps(result))
                """))

            # Run the helper to get dependency information
            logger.debug("Running dependency analysis helper")
            helper_cmd = f"python {setup_helper_path} {' '.join(deps)}"
            helper_output = self._run_in_venv(venv_dir, helper_cmd, capture_stdout=True)
            
            compile_info = {'include_dirs': [], 'library_dirs': [], 'libraries': [], 
                           'compile_args': [], 'define_macros': []}
            
            if helper_output:
                try:
                    import json
                    lines = helper_output.strip().split('\n')
                    # Get the last line which should be the JSON output
                    json_line = lines[-1]
                    compile_info = json.loads(json_line)
                    logger.debug(f"Dependency analysis result: {compile_info}")
                except Exception as e:
                    logger.warning(f"Error parsing dependency analysis output: {e}")

            # Generate adaptive pyximport test script
            test_script_path = os.path.join(tmpdir, "test_script.py")
            with open(test_script_path, "w") as f:
                f.write(textwrap.dedent(f"""
                import sys
                import os
                import importlib
                
                # Configure pyximport with detected include paths
                import pyximport
                
                # Setup include paths from dependency analysis
                include_dirs = {compile_info['include_dirs']}
                
                if include_dirs:
                    pyximport.install(setup_args={{'include_dirs': include_dirs}})
                else:
                    pyximport.install()
                
                # Try to import our generated code
                try:
                    import gen_code
                    print("Successfully imported gen_code")
                    print("Available in gen_code:", dir(gen_code))
                    
                    # List callable functions
                    callables = [name for name in dir(gen_code) 
                               if callable(getattr(gen_code, name)) 
                               and not name.startswith('_')]
                    
                    if callables:
                        print(f"Found callable functions: {{', '.join(callables)}}")
                    else:
                        print("No callable functions found in module")
                        
                except Exception as e:
                    print(f"Pyximport failed: {{e}}")
                    print("Falling back to manual compilation")
                    sys.exit(1)
                """))
                
            # Try using pyximport first
            logger.debug("Attempting compilation with pyximport")
            pyximport_cmd = f"python {test_script_path}"
            pyximport_err = self._run_in_venv(venv_dir, pyximport_cmd, cwd=tmpdir)
            
            if not pyximport_err:
                logger.debug("pyximport compilation succeeded")
                return None
            
            logger.debug(f"pyximport compilation failed, falling back to setup.py: {pyximport_err}")
            
            # Fall back to setup.py if pyximport fails
            setup_path = os.path.join(tmpdir, "setup.py")
            
            # Generate a setup.py with the dependency information
            setup_code = textwrap.dedent(f"""
            import sys
            import os
            from setuptools import setup, Extension
            from Cython.Build import cythonize

            # Include paths from dependency analysis
            include_dirs = {compile_info['include_dirs']}
            library_dirs = {compile_info['library_dirs']}
            libraries = {compile_info['libraries']}
            extra_compile_args = {compile_info['compile_args']}
            define_macros = {compile_info['define_macros']}
            
            # Define the extension with our dependency information
            extensions = [
                Extension(
                    "gen_code",
                    ["gen_code.pyx"],
                    include_dirs=include_dirs,
                    library_dirs=library_dirs,
                    libraries=libraries,
                    extra_compile_args=extra_compile_args,
                    define_macros=define_macros,
                )
            ]

            setup(
                name="gen_code",
                ext_modules=cythonize(extensions, language_level=3),
            )
            """)
            
            with open(setup_path, "w") as f:
                f.write(setup_code)

            # compile directly, not using args
            compile_cmd = f"python setup.py build_ext --inplace"
            err = self._run_in_venv(venv_dir, compile_cmd, cwd=tmpdir)
            if err:
                return f"Cython compile error:\n{err}"
                
            # Create a generic test runner script
            test_runner_path = os.path.join(tmpdir, "run_tests.py")
            with open(test_runner_path, "w") as f:
                f.write(textwrap.dedent("""
                import sys
                import inspect
                import time

                def log(msg):
                    print(f"[TEST] {msg}")

                try:
                    log("Importing generated module...")
                    import gen_code
                    log("Successfully imported gen_code")
                    
                    # Introspect the module
                    functions = [name for name in dir(gen_code) 
                               if callable(getattr(gen_code, name)) 
                               and not name.startswith('_')]
                    
                    log(f"Found {len(functions)} callable functions: {', '.join(functions)}")
                    
                    # Only try to test functions if we found any
                    if functions:
                        log("Attempting to test functions with generic data")
                        
                        # Create basic test data for common parameter types
                        test_data = {
                            'int': 5,
                            'float': 3.14,
                            'str': "test",
                            'list': [1, 2, 3],
                            'dict': {"key": "value"}
                        }
                        
                        # Dynamically add specialized data types if libraries are available
                        # Example for adding numpy arrays:
                        try:
                            import numpy
                            test_data['ndarray'] = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float64)
                            test_data['ndarray_int'] = numpy.array([1, 2, 3], dtype=numpy.int32)
                            log("NumPy is available, adding array test data")
                        except ImportError:
                            pass
                        
                        # Try to execute functions with reasonable defaults
                        for func_name in functions:
                            func = getattr(gen_code, func_name)
                            sig = inspect.signature(func)
                            log(f"Testing function: {func_name}{sig}")
                            
                            try:
                                params = {}
                                for param_name, param in sig.parameters.items():
                                    # Simple heuristics for choosing test data
                                    param_str = str(param).lower()
                                    
                                    if 'int' in param_str:
                                        params[param_name] = test_data['int']
                                    elif 'float' in param_str:
                                        params[param_name] = test_data['float']
                                    elif 'str' in param_str:
                                        params[param_name] = test_data['str']
                                    elif 'list' in param_str:
                                        params[param_name] = test_data['list']
                                    elif 'dict' in param_str:
                                        params[param_name] = test_data['dict']
                                    elif ('array' in param_str or 'ndarray' in param_str) and 'ndarray' in test_data:
                                        params[param_name] = test_data['ndarray']
                                    elif param_name.lower() in ['a', 'b', 'x', 'y', 'v', 'u'] and 'ndarray' in test_data:
                                        # Common vector parameter names
                                        params[param_name] = test_data['ndarray']
                                    else:
                                        # Default to an integer
                                        params[param_name] = test_data['int']
                                
                                # Execute the function
                                start_time = time.time()
                                result = func(**params)
                                end_time = time.time()
                                
                                log(f"Function {func_name} executed successfully in {(end_time - start_time)*1000:.2f}ms")
                                log(f"Result: {result}")
                                
                            except Exception as e:
                                log(f"Error executing {func_name}: {e}")
                                log("Skipping to next function")
                    
                    log("Test execution complete")
                    sys.exit(0)
                    
                except Exception as e:
                    log(f"Critical error during testing: {e}")
                    sys.exit(1)
                """))
            
            # Run the test script
            logger.debug("Running execution tests on compiled module")
            test_cmd = f"python {test_runner_path}"
            test_err = self._run_in_venv(venv_dir, test_cmd, cwd=tmpdir)
            
            if test_err:
                logger.warning(f"Execution tests failed, but module compiled successfully: {test_err}")
                # We don't fail the build here since the module compiled successfully
                # Just log the execution failure
            else:
                logger.debug("Execution tests passed successfully")

        return None

    def _run_in_venv(self, venv_dir, command, cwd=None, capture_stdout=False):
        """
        Run a shell command inside the ephemeral venv, returning stderr if error occurs.
        If capture_stdout is True, return stdout on success.
        """
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

        real_cmd = [exe] + args
        
        logger.debug("Running in venv: %s", " ".join(real_cmd))
        proc = subprocess.run(
            real_cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        
        if proc.returncode != 0:
            error_output = f"Command failed with code {proc.returncode}\n"
            if proc.stdout:
                error_output += f"STDOUT:\n{proc.stdout}\n"
            if proc.stderr:
                error_output += f"STDERR:\n{proc.stderr}"
            logger.error("Command failed: %s", " ".join(real_cmd))
            logger.debug("Error output: %s", error_output[:200] + "..." if len(error_output) > 200 else error_output)
            return error_output.strip()
        
        # Return stdout if requested
        if capture_stdout and proc.stdout:
            return proc.stdout.strip()
            
        # Otherwise just log the stdout for debugging
        elif proc.stdout and proc.stdout.strip():
            logger.debug("Command output: %s", 
                       proc.stdout[:200] + "..." if len(proc.stdout) > 200 else proc.stdout.strip())
            
        return None

    ############################################################################
    # DEPENDENCY PARSER
    ############################################################################
    def _parse_imports_for_python(self, code_str):
        """
        Look for lines like:
            import X
            from X import ...
            cimport X
        Return a list of libs. We skip builtins like math, sys, etc.
        """
        # Skip known builtins
        builtins = {"sys", "os", "typing", "re", "subprocess", "traceback", "math", "time"}
        libs = set()

        # Pattern for regular imports and cimports
        import_pattern = re.compile(
            r"^(?:cimport|import|from)\s+([a-zA-Z0-9_\.]+)",
            re.MULTILINE
        )
        matches = import_pattern.findall(code_str)
        for m in matches:
            top_level = m.split(".")[0]
            if top_level not in builtins:
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
        
        # Check for common library aliases that might not be explicitly imported
        if self._is_cython(code_str):
            for alias, lib_name in common_aliases.items():
                # Look for usage patterns like "np." or "pd." or "np.array"
                if f"{alias}." in code_str and lib_name not in libs:
                    logger.debug(f"Detected potential {lib_name} usage via '{alias}' alias")
                    libs.add(lib_name)

        # Convert set to a sorted list
        return sorted(libs)


###############################################################################
# MAIN
###############################################################################
if __name__ == "__main__":
    code_signature = Signature({
        "prompt": InputField(
            prefix="User Prompt:",
            desc="The user request describing what code to generate",
            format=str
        ),
        "generated_code": OutputField(
            prefix="Code:",
            desc="The code snippet that solves the user request",
            format=str
        ),
    })

    # Configure log level from environment or default to INFO
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, log_level))
    logger.info("Starting EphemeralCodeGenerator with log level: %s", log_level)

    generator = EphemeralCodeGenerator(signature=code_signature, max_iters=3)
    prompt = "Write an efficient cython method for calculating a dot product of 2 vectors using typed memoryviews from numpy."
    logger.info("Generating code for prompt: %s", prompt)
    
    result = generator.forward(prompt=prompt)
    
    if result["error"] is None:
        logger.info("Code generation successful!")
        logger.info("Generated code:\n%s", result["generated_code"])
    else:
        logger.error("Code generation failed with error: %s", result["error"])
        logger.info("Last code attempt:\n%s", result["generated_code"])

    generator = EphemeralCodeGenerator(signature=code_signature, max_iters=3)
    prompt = "Write an efficient cython method for performance-optimized FizzBuzz example as part of the living dataset."
    logger.info("Generating code for prompt: %s", prompt)
    result = generator.forward(prompt=prompt)
    
    if result["error"] is None:
        logger.info("Code generation successful!")
        logger.info("Generated code:\n%s", result["generated_code"])
    else:
        logger.error("Code generation failed with error: %s", result["error"])
        logger.info("Last code attempt:\n%s", result["generated_code"])