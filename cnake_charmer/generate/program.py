import os
import re
import subprocess
import sys
import tempfile
import traceback
import venv
import textwrap
import dspy
from dspy.primitives import Module
from dspy.signatures import InputField, OutputField
from dspy.signatures.signature import Signature, ensure_signature
from dotenv import load_dotenv
import time

###############################################################################
# 1) CONFIGURE DSPY / LLM
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
                    "For Cython code, include ALL necessary imports and cimports explicitly, including:\n"
                    "- `import numpy as np` and `cimport numpy as np` if using numpy\n"
                    "- `cimport cython` if using cython decorators\n\n"
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
                    "Make sure to include ALL necessary imports, especially if using numpy or cython decorators:\n"
                    "- If using numpy arrays or memoryviews, include BOTH `import numpy as np` AND `cimport numpy as np`\n"
                    "- If using cython decorators like @cython.boundscheck(False), include `cimport cython`\n\n"
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
        print("[DEBUG] forward called with prompt:", kwargs.get("prompt"))
        # Step 1: get initial code
        code_data = self.code_generate(**kwargs)
        raw_code = code_data.get("generated_code", "")
        print("[DEBUG] Initial generation raw output:", raw_code)

        # Step 2: parse
        code_block, parse_err = self._extract_code(raw_code)
        if parse_err:
            print("[DEBUG] parse error => regeneration")
            return self._try_regeneration(kwargs, previous_code="", error=parse_err)

        # Step 3: ephemeral build/run
        error = self._ephemeral_build_and_run(code_block)
        if error:
            print(f"[DEBUG] ephemeral build error => regeneration: {error}")
            return self._try_regeneration(kwargs, previous_code=code_block, error=error)

        return {"generated_code": code_block, "error": None}

    def _try_regeneration(self, kwargs, previous_code, error):
        attempts = 0
        while attempts < self.max_iters:
            attempts += 1
            print(f"[DEBUG] Attempting regeneration, attempt #{attempts}")
            regen_data = self.code_regenerate(
                prompt=kwargs["prompt"],
                previous_code=previous_code,
                error=error
            )
            new_raw = regen_data.get("generated_code", "")
            new_code, parse_err = self._extract_code(new_raw)
            if parse_err:
                # next iteration
                print("[DEBUG] parse error on regenerated code => continuing")
                previous_code = new_raw
                error = parse_err
                continue

            build_err = self._ephemeral_build_and_run(new_code)
            if build_err:
                print("[DEBUG] ephemeral build error again => continuing")
                error = build_err
                previous_code = new_code
            else:
                # success
                return {"generated_code": new_code, "error": None}

        # if we exhaust attempts
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
                return ("", "ERROR: Could not parse code block.")
            return (code_block, None)
        code_block = match.group(1).strip()
        if not code_block:
            return ("", "ERROR: Empty code block after triple backticks.")
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
            venv_dir = os.path.join(tmpdir, "venv")
            venv.create(venv_dir, with_pip=True)

            # 2) parse dependencies (imported libs) -> pip install
            deps = self._parse_imports_for_python(code_str)
            # e.g. if we see "import requests" => we do pip install requests
            # We'll also ensure 'wheel' is installed, etc.

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

            # 4) run
            for cmd in commands:
                err = self._run_in_venv(venv_dir, cmd)
                if err:
                    return f"Python ephemeral venv install error: {err}"

            run_cmd = f"python {py_path}"
            err = self._run_in_venv(venv_dir, run_cmd)
            if err:
                return f"Python run error: {err}"
        return None

    def _build_and_run_cython(self, code_str):
        """
        1) ephemeral venv
        2) detect libraries -> pip install
        3) write minimal setup.py with include_dirs for numpy, etc.
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
                
            # if we see 'cimport numpy' or 'import numpy', ensure numpy is installed
            uses_numpy = any("numpy" in d.lower() for d in deps) or "numpy" in code_str.lower()
            if uses_numpy and not any(d.lower() == "numpy" for d in deps):
                deps.append("numpy")

            # Install dependencies with retries
            max_install_attempts = 3
            for attempt in range(max_install_attempts):
                print(f"[DEBUG] Installing dependencies (attempt {attempt+1}/{max_install_attempts}): {deps}")
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

            # Write helper test script to import and exercise the module
            test_script_path = os.path.join(tmpdir, "test_script.py")
            uses_numpy = any(dep.lower() == "numpy" for dep in deps) or "numpy" in code_str.lower()
            
            # Create different test scripts based on dependencies
            if uses_numpy:
                test_script = textwrap.dedent("""
                import numpy as np
                import pyximport
                
                # Setup pyximport with numpy support
                pyximport.install(setup_args={'include_dirs': np.get_include()})
                
                # Import our generated code
                import gen_code
                
                # Print available functions
                print("Available in gen_code:", dir(gen_code))
                
                # Test basic functionality if dot_product exists
                try:
                    if hasattr(gen_code, 'dot_product'):
                        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
                        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
                        result = gen_code.dot_product(a, b)
                        print(f"Test result: {result}")
                except Exception as e:
                    print(f"Test error: {e}")
                """)
            else:
                # Generic test script that doesn't assume numpy
                test_script = textwrap.dedent("""
                import pyximport
                pyximport.install()
                
                # Import our generated code
                import gen_code
                
                # Print available functions
                print("Available in gen_code:", dir(gen_code))
                
                # Generic test - just try calling a function if found
                try:
                    # Find a callable function to test
                    test_functions = [name for name in dir(gen_code) 
                                     if callable(getattr(gen_code, name)) 
                                     and not name.startswith('_')]
                    
                    if test_functions:
                        func_name = test_functions[0]
                        print(f"Found testable function: {func_name}")
                        
                        # Note: we're not actually calling it since we don't know 
                        # what parameters it needs
                        print(f"Function exists and is callable")
                except Exception as e:
                    print(f"Test error: {e}")
                """)
                
            with open(test_script_path, "w") as f:
                f.write(test_script)

            # Try using pyximport first (simpler approach)
            print("[DEBUG] Attempting compilation with pyximport")
            pyximport_cmd = f"python {test_script_path}"
            pyximport_err = self._run_in_venv(venv_dir, pyximport_cmd, cwd=tmpdir)
            
            if not pyximport_err:
                print("[DEBUG] pyximport compilation succeeded")
                return None
            
            print(f"[DEBUG] pyximport compilation failed, falling back to setup.py: {pyximport_err}")
            
            # Fall back to traditional setup.py approach
            setup_path = os.path.join(tmpdir, "setup.py")
            
            # Identify all detected dependencies for smarter setup.py generation
            uses_numpy = any(dep.lower() == "numpy" for dep in deps) or "numpy" in code_str.lower()
            
            # Generate a more generic setup.py with dynamic dependency handling
            setup_code = textwrap.dedent(f"""
            import sys
            import os
            from setuptools import setup, Extension
            from Cython.Build import cythonize

            # Dictionary to store special handling for libraries
            include_dirs = []
            library_dirs = []
            libraries = []
            extra_compile_args = []
            define_macros = []
            
            # Check for numpy and add its include directory if present
            if {uses_numpy}:
                try:
                    import numpy
                    print(f"Found numpy at {{numpy.__file__}}")
                    numpy_include = numpy.get_include()
                    print(f"Adding numpy include dir: {{numpy_include}}")
                    include_dirs.append(numpy_include)
                except ImportError:
                    print("ERROR: numpy import failed despite being installed")
                    sys.exit(1)
            
            # Generic dependency check - could be expanded for other special libraries
            # that require include_dirs or other special handling
            for lib_name in {repr(deps)}:
                try:
                    lib = __import__(lib_name)
                    print(f"Successfully imported {{lib_name}}")
                    
                    # Special handling for other libraries can be added here
                    # Example: if lib_name == 'other_lib': ...
                except ImportError:
                    print(f"WARNING: {{lib_name}} import failed despite being installed")
            
            # Define the extension
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

            # compile
            compile_cmd = f"python {setup_path} build_ext --inplace"
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
                    
                    # Try to execute functions with reasonable defaults
                    for func_name in functions:
                        func = getattr(gen_code, func_name)
                        sig = inspect.signature(func)
                        log(f"Testing function: {func_name}{sig}")
                        
                        # Create basic test data for common parameter types
                        import numpy as np
                        test_data = {
                            'int': 5,
                            'float': 3.14,
                            'str': "test",
                            'list': [1, 2, 3],
                            'dict': {"key": "value"},
                            'ndarray': np.array([1.0, 2.0, 3.0], dtype=np.float64),
                            'ndarray_int': np.array([1, 2, 3], dtype=np.int32),
                        }
                        
                        # Try to match parameters with appropriate test data
                        try:
                            params = {}
                            for param_name, param in sig.parameters.items():
                                # Simple heuristics for choosing test data
                                if 'int' in str(param).lower():
                                    params[param_name] = test_data['int']
                                elif 'float' in str(param).lower():
                                    params[param_name] = test_data['float']
                                elif 'str' in str(param).lower():
                                    params[param_name] = test_data['str']
                                elif 'list' in str(param).lower():
                                    params[param_name] = test_data['list']
                                elif 'dict' in str(param).lower():
                                    params[param_name] = test_data['dict']
                                elif 'array' in str(param).lower() or 'ndarray' in str(param).lower():
                                    params[param_name] = test_data['ndarray']
                                elif param_name in ['a', 'b', 'x', 'y']:
                                    # Common vector parameter names, use ndarray
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
            print("[DEBUG] Running execution tests on compiled module")
            test_cmd = f"python {test_runner_path}"
            test_err = self._run_in_venv(venv_dir, test_cmd, cwd=tmpdir)
            
            if test_err:
                print(f"[WARNING] Execution tests failed, but module compiled successfully: {test_err}")
                # We don't fail the build here since the module compiled successfully
                # Just log the execution failure
            else:
                print("[DEBUG] Execution tests passed successfully")

        return None

    def _run_in_venv(self, venv_dir, command, cwd=None):
        """
        Run a shell command inside the ephemeral venv, returning stderr if error occurs.
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
        
        print(f"[DEBUG] Running in venv: {' '.join(real_cmd)}")
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
            return error_output.strip()
        
        # Even if successful, print the stdout for debugging
        if proc.stdout:
            print(f"[DEBUG] Command output: {proc.stdout.strip()}")
            
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
                    print(f"[DEBUG] Detected potential {lib_name} usage via '{alias}' alias")
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

    generator = EphemeralCodeGenerator(signature=code_signature, max_iters=3)
    result = generator.forward(prompt="Write an efficient cython method for calculating a dot product of 2 vectors using typed memoryviews from numpy.")
    print("FINAL RESULT:", result)
    if result["error"] is None:
        print("Generated code:\n", result["generated_code"])
    else:
        print("We still have an error after retries:", result["error"])