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
import json
from dspy.primitives import Module
from dspy.signatures import InputField, OutputField
from dspy.signatures.signature import Signature, ensure_signature
from dotenv import load_dotenv

###############################################################################
# 1) ENHANCED LOGGING CONFIGURATION
###############################################################################
# Configure logger with more detailed formatting
logger = logging.getLogger("EphemeralCodeGenerator")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)  # Set to logging.DEBUG for more verbose output

# Create a file handler to capture logs to a file in addition to console
try:
    file_handler = logging.FileHandler("code_generator.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("File logging initialized to code_generator.log")
except Exception as e:
    logger.warning(f"Could not initialize file logging: {str(e)}")

# Add debug context information to log messages
def log_context(message, context=None):
    """Add context information to log messages for better debugging"""
    if context:
        return f"{message} | Context: {context}"
    return message

###############################################################################
# 2) CONFIGURE DSPY / LLM
###############################################################################
load_dotenv()
# Simple configuration with OpenRouter
lm = dspy.LM(model="openrouter/anthropic/claude-3.7-sonnet")
dspy.configure(lm=lm)
logger.info(f"Configured DSPy with LM: {lm.__class__.__name__}")

###############################################################################
# 3) CODE GENERATOR CLASS
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
        logger.info(f"Initialized EphemeralCodeGenerator with max_iters={max_iters}")

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
                    "Your response must be enclosed in triple backticks (```), with NO language indicator after the opening backticks. "
                    "Include ONLY the code itself, no additional commentary before or after the code block.\n\n"
                    
                    "Code quality requirements:\n"
                    "1. Follow PEP 8 style guidelines (proper spacing, naming conventions, max line length of 79 characters)\n"
                    "2. Include Google-style docstrings for all functions, classes, and modules\n"
                    "3. Add appropriate comments for complex logic\n\n"
                    
                    "For Cython code:\n"
                    "- Add comment-based directives at the top of the file to optimize performance:\n"
                    "  # cython: boundscheck=False\n"
                    "  # cython: wraparound=False\n"
                    "- Include ALL necessary imports and cimports explicitly\n"
                    "- Add proper type declarations for all functions and variables\n"
                    "- Remember that Python standard library modules (collections, threading, etc.) don't need external installation\n\n"
                )
            ),
        )
        logger.debug("Initialized code_generate chain")

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
        logger.debug("Initialized code_regenerate chain")

    def forward(self, **kwargs):
        """
        1) Generate code.
        2) Parse code block from triple backticks.
        3) Attempt ephemeral build/run.
        4) If error => regeneration loop up to max_iters.
        """
        request_id = id(kwargs)
        logger.info(log_context(f"Forward called with prompt [ID: {request_id}]: {kwargs.get('prompt', '')[:100]}...", 
                              {"request_id": request_id}))
        
        # Step 1: get initial code
        try:
            logger.debug(f"Request {request_id}: Calling code_generate")
            code_data = self.code_generate(**kwargs)
            logger.debug(f"Request {request_id}: code_generate returned {type(code_data)}")
            
            raw_code = ""
            if hasattr(code_data, 'generated_code'):
                raw_code = code_data.generated_code
                logger.debug(f"Request {request_id}: Extracted code from Prediction object")
            else:
                raw_code = code_data.get("generated_code", "")
                logger.debug(f"Request {request_id}: Extracted code from dictionary")
                
            logger.debug(log_context(f"Request {request_id}: Initial generation raw output: {raw_code[:200]}...", 
                                   {"raw_output_length": len(raw_code)}))
        except Exception as e:
            logger.error(f"Request {request_id}: Initial code generation failed: {str(e)}")
            logger.debug(f"Request {request_id}: Generation error details: {traceback.format_exc()}")
            return {"generated_code": "", "error": f"Code generation failed: {str(e)}"}

        # Step 2: parse
        code_block, parse_err = self._extract_code(raw_code, request_id)
        if parse_err:
            logger.warning(log_context(f"Request {request_id}: Parse error => regeneration: {parse_err}", 
                                      {"parse_error": parse_err}))
            return self._try_regeneration(kwargs, previous_code="", error=parse_err, request_id=request_id)

        # Step 3: ephemeral build/run
        error = self._ephemeral_build_and_run(code_block, request_id)
        if error:
            logger.warning(log_context(f"Request {request_id}: Ephemeral build error => regeneration: {error[:1000]}...", 
                                      {"error_length": len(error)}))
            return self._try_regeneration(kwargs, previous_code=code_block, error=error, request_id=request_id)

        logger.info(f"Request {request_id}: Successfully generated and built code")
        return {"generated_code": code_block, "error": None}

    def _try_regeneration(self, kwargs, previous_code, error, request_id=None):
        """
        Regeneration loop with improved logging
        """
        if request_id is None:
            request_id = id(kwargs)
            
        attempts = 0
        while attempts < self.max_iters:
            attempts += 1
            logger.info(f"Request {request_id}: Attempting regeneration, attempt #{attempts}/{self.max_iters}")
            
            # Log the inputs to regeneration for debugging
            logger.debug(f"Request {request_id}: Regeneration input prompt: {kwargs.get('prompt', '')[:50]}...")
            logger.debug(f"Request {request_id}: Regeneration previous code length: {len(previous_code)}")
            logger.debug(f"Request {request_id}: Regeneration error: {error[:100]}..." if len(error) > 100 else error)
            
            try:
                regen_data = self.code_regenerate(
                    prompt=kwargs["prompt"],
                    previous_code=previous_code,
                    error=error
                )
                
                # Handle Prediction objects from DSPy
                logger.debug(f"Request {request_id}: Regeneration returned type: {type(regen_data)}")
                
                new_raw = ""
                if hasattr(regen_data, 'generated_code'):
                    new_raw = regen_data.generated_code
                    logger.debug(f"Request {request_id}: Used generated_code attribute")
                else:
                    new_raw = regen_data.get("generated_code", "")
                    logger.debug(f"Request {request_id}: Used dictionary access")
                
                logger.debug(f"Request {request_id}: Regenerated code length: {len(new_raw)}")
                
            except Exception as e:
                logger.error(f"Request {request_id}: Regeneration attempt #{attempts} failed: {str(e)}")
                logger.debug(f"Request {request_id}: Regeneration error details: {traceback.format_exc()}")
                continue
            
            new_code, parse_err = self._extract_code(new_raw, request_id)
            if parse_err:
                # next iteration
                logger.warning(log_context(
                    f"Request {request_id}: Parse error on regenerated code (attempt #{attempts}) => continuing: {parse_err}",
                    {"parse_error": parse_err}
                ))
                previous_code = new_raw
                error = parse_err
                continue

            build_err = self._ephemeral_build_and_run(new_code, request_id)
            if build_err:
                logger.warning(log_context(
                    f"Request {request_id}: Ephemeral build error again (attempt #{attempts}) => continuing: {build_err[:300]}...",
                    {"error_length": len(build_err)}
                ))
                error = build_err
                previous_code = new_code
            else:
                # success
                logger.info(f"Request {request_id}: Regeneration successful on attempt #{attempts}")
                return {"generated_code": new_code, "error": None}

        # if we exhaust attempts
        logger.error(f"Request {request_id}: Exhausted all {self.max_iters} regeneration attempts, still has error")
        return {"generated_code": previous_code, "error": error}

    ############################################################################
    # CODE PARSING
    ############################################################################

    def _extract_code(self, text, request_id=None):
        """
        Grab triple-backtick code from LLM response. If missing, fallback to entire text.
        """
        if request_id is None:
            request_id = id(text)
            
        logger.debug(f"Request {request_id}: Extracting code from text of length {len(text)}")
        
        # Handle empty or None text
        if not text:
            logger.error(f"Request {request_id}: Empty text input to code extraction")
            return ("", "ERROR: Empty code text input.")
            
        try:
            match = re.search(r"```[\w\s]*\n?(.*?)```", text, re.DOTALL)
            if not match:
                logger.debug(f"Request {request_id}: No triple backticks found, first 100 chars: {text[:100]}...")
                code_block = text.strip()
                if not code_block:
                    logger.error(f"Request {request_id}: Could not parse code block - empty content")
                    return ("", "ERROR: Could not parse code block.")
                logger.warning(f"Request {request_id}: No triple backticks found, using entire text as code")
                return (code_block, None)
            
            code_block = match.group(1).strip()
            if not code_block:
                logger.error(f"Request {request_id}: Empty code block after triple backticks")
                return ("", "ERROR: Empty code block after triple backticks.")
            
            # Check if we need to handle multiple code blocks
            all_code_blocks = re.findall(r"```[\w\s]*\n?(.*?)```", text, re.DOTALL)
            if len(all_code_blocks) > 1:
                logger.info(f"Request {request_id}: Found {len(all_code_blocks)} code blocks, using the first one")
                
            logger.info(log_context(
                f"Request {request_id}: Successfully extracted code block ({len(code_block)} characters)",
                {"num_blocks": len(all_code_blocks) if 'all_code_blocks' in locals() else 1}
            ))
            
            logger.debug(f"Request {request_id}: Code block begins with: {code_block[:100]}..." 
                      if len(code_block) > 100 else code_block)
            return (code_block, None)
        except Exception as e:
            logger.error(f"Request {request_id}: Exception during code extraction: {str(e)}")
            logger.debug(f"Request {request_id}: Extraction error details: {traceback.format_exc()}")
            return ("", f"ERROR: Code extraction failed: {str(e)}")

    ############################################################################
    # EPHEMERAL BUILD + RUN
    ############################################################################

    def _ephemeral_build_and_run(self, code_str, request_id=None):
        """
        1) Create ephemeral venv
        2) Detect imports => pip install them (plus Cython if needed)
        3) If Cython => attempt compile. Otherwise => run Python.
        Return error string or None on success.
        """
        if request_id is None:
            request_id = id(code_str)
            
        # Are we dealing with Cython or Python?
        is_cython = self._is_cython(code_str)
        logger.info(f"Request {request_id}: Code identified as {'Cython' if is_cython else 'Python'}")
        
        if is_cython:
            return self._build_and_run_cython(code_str, request_id)
        else:
            return self._build_and_run_python(code_str, request_id)

    def _is_cython(self, code_str):
        """Check if code is Cython based on key indicators"""
        low = code_str.lower()
        cython_indicators = ["cdef", ".pyx", "cimport", "cython"]
        
        # Check each indicator and log which ones were found
        found_indicators = [ind for ind in cython_indicators if ind in (code_str if ind != "cython" else low)]
        
        if found_indicators:
            logger.debug(f"Identified as Cython due to: {', '.join(found_indicators)}")
            return True
        return False

    def _build_and_run_python(self, code_str, request_id=None):
        """
        Ephemeral venv -> install detected libs -> run code with python.
        """
        if request_id is None:
            request_id = id(code_str)
            
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1) create ephemeral venv
            logger.info(f"Request {request_id}: Creating ephemeral venv for Python execution in {tmpdir}")
            venv_dir = os.path.join(tmpdir, "venv")
            try:
                venv.create(venv_dir, with_pip=True)
                logger.debug(f"Request {request_id}: Successfully created venv at {venv_dir}")
            except Exception as e:
                logger.error(f"Request {request_id}: Error creating venv: {str(e)}")
                return f"Failed to create virtual environment: {str(e)}"

            # 2) parse dependencies (imported libs) -> pip install
            try:
                deps = self._parse_imports_for_python(code_str)
                logger.info(log_context(
                    f"Request {request_id}: Detected dependencies: {deps}",
                    {"num_dependencies": len(deps)}
                ))
            except Exception as e:
                logger.error(f"Request {request_id}: Error parsing dependencies: {str(e)}")
                return f"Failed to parse dependencies: {str(e)}"

            commands = [
                # upgrade pip
                f"pip install --upgrade pip wheel setuptools",
            ]
            if deps:
                commands.append(f"pip install {' '.join(deps)}")

            # 3) write code to .py
            py_path = os.path.join(tmpdir, "gen_code.py")
            try:
                with open(py_path, "w") as f:
                    f.write(code_str)
                logger.info(f"Request {request_id}: Wrote Python code ({len(code_str)} bytes) to {py_path}")
            except Exception as e:
                logger.error(f"Request {request_id}: Error writing code to file: {str(e)}")
                return f"Failed to write code to file: {str(e)}"

            # 4) run
            for i, cmd in enumerate(commands):
                logger.info(f"Request {request_id}: Running command [{i+1}/{len(commands)}]: {cmd}")
                err = self._run_in_venv(venv_dir, cmd, request_id=request_id)
                if err:
                    logger.error(log_context(
                        f"Request {request_id}: Dependency installation failed: {err[:100]}...",
                        {"error_full_length": len(err)}
                    ))
                    return f"Python ephemeral venv install error: {err}"

            logger.info(f"Request {request_id}: Executing Python code")
            run_cmd = f"python {py_path}"
            err = self._run_in_venv(venv_dir, run_cmd, request_id=request_id)
            if err:
                logger.error(log_context(
                    f"Request {request_id}: Python execution failed: {err[:100]}...",
                    {"error_full_length": len(err)}
                ))
                return f"Python run error: {err}"
                
            logger.info(f"Request {request_id}: Python execution completed successfully")
        return None

    def _build_and_run_cython(self, code_str, request_id=None):
        """
        1) ephemeral venv
        2) detect libraries -> pip install
        3) write setup.py for compilation
        4) compile + run
        """
        if request_id is None:
            request_id = id(code_str)
            
        with tempfile.TemporaryDirectory() as tmpdir:
            logger.debug(f"Request {request_id}: Created temporary directory: {tmpdir}")
            
            try:
                venv_dir = os.path.join(tmpdir, "venv")
                logger.debug(f"Request {request_id}: Creating venv at {venv_dir}")
                venv.create(venv_dir, with_pip=True)
                logger.debug(f"Request {request_id}: Successfully created venv")
            except Exception as e:
                logger.error(f"Request {request_id}: Error creating venv: {str(e)}")
                return f"Failed to create virtual environment for Cython: {str(e)}"

            # parse imports => gather needed deps
            try:
                deps = self._parse_imports_for_python(code_str)
                logger.debug(f"Request {request_id}: Parsed dependencies: {deps}")
            except Exception as e:
                logger.error(f"Request {request_id}: Error parsing dependencies: {str(e)}")
                return f"Failed to parse dependencies: {str(e)}"
            
            # always need cython
            if not any(d.lower() == "cython" for d in deps):
                deps.append("cython")
                logger.debug(f"Request {request_id}: Added cython to dependencies")

            # Install dependencies with retries
            max_install_attempts = 3
            for attempt in range(max_install_attempts):
                logger.debug(log_context(
                    f"Request {request_id}: Installing dependencies (attempt {attempt+1}/{max_install_attempts}): {deps}",
                    {"num_dependencies": len(deps)}
                ))
                
                commands = [
                    f"pip install --upgrade pip wheel setuptools",
                    f"pip install {' '.join(deps)}",
                ]
                install_error = None
                
                for i, cmd in enumerate(commands):
                    logger.debug(f"Request {request_id}: Running command [{i+1}/{len(commands)}]: {cmd}")
                    err = self._run_in_venv(venv_dir, cmd, request_id=request_id)
                    if err:
                        install_error = f"Cython ephemeral venv install error: {err}"
                        logger.warning(f"Request {request_id}: Command failed: {err[:300]}...")
                        break
                
                if not install_error:
                    logger.debug(f"Request {request_id}: Successfully installed all dependencies")
                    break
                    
                # Wait before retry
                if attempt < max_install_attempts - 1:
                    sleep_time = (attempt + 1) * 2  # Exponential backoff
                    logger.debug(f"Request {request_id}: Waiting {sleep_time}s before retry")
                    time.sleep(sleep_time)
            
            if install_error:
                logger.error(f"Request {request_id}: Failed to install dependencies after {max_install_attempts} attempts")
                return install_error

            # write the .pyx
            pyx_path = os.path.join(tmpdir, "gen_code.pyx")
            try:
                with open(pyx_path, "w") as f:
                    f.write(code_str)
                logger.debug(f"Request {request_id}: Wrote Cython code ({len(code_str)} bytes) to {pyx_path}")
            except Exception as e:
                logger.error(f"Request {request_id}: Error writing Cython code to file: {str(e)}")
                return f"Failed to write Cython code to file: {str(e)}"

            # Create a helper script to generate setup.py with proper dependency information
            setup_helper_path = os.path.join(tmpdir, "setup_helper.py")
            try:
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
                logger.debug(f"Request {request_id}: Wrote dependency helper script")
            except Exception as e:
                logger.error(f"Request {request_id}: Error writing helper script: {str(e)}")
                return f"Failed to write setup helper script: {str(e)}"

            # Run the helper to get dependency information
            logger.debug(f"Request {request_id}: Running dependency analysis helper")
            helper_cmd = f"python {setup_helper_path} {' '.join(deps)}"
            helper_output = self._run_in_venv(venv_dir, helper_cmd, capture_stdout=True, request_id=request_id)
            
            compile_info = {'include_dirs': [], 'library_dirs': [], 'libraries': [], 
                           'compile_args': [], 'define_macros': []}
            
            if helper_output:
                try:
                    lines = helper_output.strip().split('\n')
                    # Get the last line which should be the JSON output
                    json_line = lines[-1]
                    compile_info = json.loads(json_line)
                    logger.debug(log_context(
                        f"Request {request_id}: Dependency analysis result: {compile_info}",
                        {"include_dirs_count": len(compile_info['include_dirs'])}
                    ))
                except Exception as e:
                    logger.warning(log_context(
                        f"Request {request_id}: Error parsing dependency analysis output: {str(e)}",
                        {"raw_output": helper_output[:300] + "..." if len(helper_output) > 300 else helper_output}
                    ))

            # Generate adaptive pyximport test script
            test_script_path = os.path.join(tmpdir, "test_script.py")
            try:
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
                        pyximport.install(setup_args={{"include_dirs": include_dirs}})
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
                logger.debug(f"Request {request_id}: Wrote pyximport test script")
            except Exception as e:
                logger.error(f"Request {request_id}: Error writing test script: {str(e)}")
                return f"Failed to write test script: {str(e)}"
                
            # Try using pyximport first
            logger.debug(f"Request {request_id}: Attempting compilation with pyximport")
            pyximport_cmd = f"python {test_script_path}"
            pyximport_err = self._run_in_venv(venv_dir, pyximport_cmd, cwd=tmpdir, request_id=request_id)
            
            if not pyximport_err:
                logger.debug(f"Request {request_id}: pyximport compilation succeeded")
                return None
            
            logger.debug(log_context(
                f"Request {request_id}: pyximport compilation failed, falling back to setup.py",
                {"error": pyximport_err[:300] + "..." if len(pyximport_err) > 300 else pyximport_err}
            ))
            
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
            
            try:
                with open(setup_path, "w") as f:
                    f.write(setup_code)
                logger.debug(f"Request {request_id}: Wrote setup.py for Cython compilation")
            except Exception as e:
                logger.error(f"Request {request_id}: Error writing setup.py: {str(e)}")
                return f"Failed to write setup.py: {str(e)}"

            # compile directly, not using args
            logger.info(f"Request {request_id}: Compiling Cython code with setup.py")
            compile_cmd = f"python setup.py build_ext --inplace"
            err = self._run_in_venv(venv_dir, compile_cmd, cwd=tmpdir, request_id=request_id)
            if err:
                logger.error(log_context(
                    f"Request {request_id}: Cython compilation failed",
                    {"error": err[:1000] + "..." if len(err) > 1000 else err}
                ))
                return f"Cython compile error:\n{err}"
                
            logger.info(f"Request {request_id}: Cython compilation successful")
                
            # Create a generic test runner script
            test_runner_path = os.path.join(tmpdir, "run_tests.py")
            try:
                with open(test_runner_path, "w") as f:
                    f.write(textwrap.dedent("""
                    import sys
                    import inspect
                    import time

                    def log(msg):
                        print("[TEST] " + str(msg))

                    try:
                        log("Importing generated module...")
                        import gen_code
                        log("Successfully imported gen_code")
                        
                        # Introspect the module
                        functions = [name for name in dir(gen_code) 
                                   if callable(getattr(gen_code, name)) 
                                   and not name.startswith('_')]
                        
                        log("Found " + str(len(functions)) + " callable functions: " + ", ".join(functions))
                        
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
                                log("Testing function: " + func_name + str(sig))
                                
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
                                    
                                    log("Function " + func_name + " executed successfully in " + str((end_time - start_time)*1000) + "ms")
                                    log("Result: " + str(result))
                                    
                                except Exception as e:
                                    log("Error executing " + func_name + ": " + str(e))
                                    log("Skipping to next function")
                            
                        log("Test execution complete")
                        sys.exit(0)
                        
                    except Exception as e:
                        log("Critical error during testing: " + str(e))
                        sys.exit(1)
                    """))
                logger.debug(f"Request {request_id}: Wrote test runner script")
            except Exception as e:
                logger.error(f"Request {request_id}: Error writing test runner: {str(e)}")
                # Continue despite this error since it's just testing
            
            # Run the test script
            logger.debug(f"Request {request_id}: Running execution tests on compiled module")
            test_cmd = f"python {test_runner_path}"
            test_err = self._run_in_venv(venv_dir, test_cmd, cwd=tmpdir, request_id=request_id)
            
            if test_err:
                logger.warning(log_context(
                    f"Request {request_id}: Execution tests failed, but module compiled successfully",
                    {"error": test_err[:300] + "..." if len(test_err) > 300 else test_err}
                ))
                # We don't fail the build here since the module compiled successfully
                # Just log the execution failure
            else:
                logger.debug(f"Request {request_id}: Execution tests passed successfully")

        return None

    def _run_in_venv(self, venv_dir, command, cwd=None, capture_stdout=False, request_id=None):
        """
        Run a shell command inside the ephemeral venv, returning stderr if error occurs.
        If capture_stdout is True, return stdout on success.
        """
        if request_id is None:
            request_id = id(command)
            
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
        cmd_str = " ".join(real_cmd)
        
        # Check if the executable exists
        if not os.path.exists(exe):
            logger.error(f"Request {request_id}: Executable not found: {exe}")
            return f"Command execution failed: executable not found at {exe}"
        
        logger.debug(log_context(
            f"Request {request_id}: Running in venv: {cmd_str}",
            {"cwd": cwd, "capture_stdout": capture_stdout}
        ))
        
        try:
            proc = subprocess.run(
                real_cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
            )
            
            # Always log some basic info about the execution
            logger.debug(f"Request {request_id}: Command returned with code {proc.returncode}")
            
            if proc.returncode != 0:
                error_output = f"Command failed with code {proc.returncode}\n"
                if proc.stdout:
                    error_output += f"STDOUT:\n{proc.stdout}\n"
                if proc.stderr:
                    error_output += f"STDERR:\n{proc.stderr}"
                logger.error(log_context(
                    f"Request {request_id}: Command failed: {cmd_str}",
                    {"returncode": proc.returncode}
                ))
                logger.debug(log_context(
                    f"Request {request_id}: Error output",
                    {"stdout_len": len(proc.stdout) if proc.stdout else 0, 
                     "stderr_len": len(proc.stderr) if proc.stderr else 0}
                ))
                return error_output.strip()
            
            # Return stdout if requested
            if capture_stdout and proc.stdout:
                logger.debug(f"Request {request_id}: Command captured stdout ({len(proc.stdout)} chars)")
                return proc.stdout.strip()
                
            # Otherwise just log the stdout for debugging
            elif proc.stdout and proc.stdout.strip():
                logger.debug(log_context(
                    f"Request {request_id}: Command output",
                    {"stdout_len": len(proc.stdout), 
                     "stdout_snippet": proc.stdout[:300] + "..." if len(proc.stdout) > 300 else proc.stdout.strip()}
                ))
                
        except Exception as e:
            logger.error(f"Request {request_id}: Exception executing command: {str(e)}")
            logger.debug(f"Request {request_id}: Execution error details: {traceback.format_exc()}")
            return f"Command execution failed with exception: {str(e)}"
            
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
        # Skip known builtins (Python standard library modules)
        builtins = {
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
        system_libs = {"libc", "cpython", "libcpp", "posix"}
        
        libs = set()

        try:
            # Pattern for regular imports and cimports
            import_pattern = re.compile(
                r"^(?:cimport|import|from)\s+([a-zA-Z0-9_\.]+)",
                re.MULTILINE
            )
            matches = import_pattern.findall(code_str)
            logger.debug(f"Found {len(matches)} import statements in code")
            
            for m in matches:
                top_level = m.split(".")[0]
                if top_level not in builtins and top_level not in system_libs:
                    libs.add(top_level)
                    logger.debug(f"Added '{top_level}' to dependencies")

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
        except Exception as e:
            logger.error(f"Error parsing dependencies: {str(e)}")
            logger.debug(f"Dependency parsing error details: {traceback.format_exc()}")
            # Return a minimal set to avoid complete failure
            return ["cython"] if self._is_cython(code_str) else []

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

    # Capture uncaught exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Let keyboard interrupts pass through
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    # Install the exception handler
    sys.excepthook = handle_exception
    logger.info("Installed global exception handler")
    
    generator = EphemeralCodeGenerator(signature=code_signature, max_iters=3)
    prompt = "Write an efficient cython method for calculating a dot product of 2 vectors using typed memoryviews from numpy."
    logger.info("Generating code for prompt: %s", prompt)
    
    try:
        result = generator.forward(prompt=prompt)
        
        if result["error"] is None:
            logger.info("Code generation successful!")
            logger.info("Generated code:\n%s", result["generated_code"])
        else:
            logger.error("Code generation failed with error: %s", result["error"])
            logger.info("Last code attempt:\n%s", result["generated_code"])
    except Exception as e:
        logger.critical(f"Unhandled exception in code generation: {str(e)}")
        logger.debug(f"Exception details: {traceback.format_exc()}")

    try:
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
    except Exception as e:
        logger.critical(f"Unhandled exception in second code generation: {str(e)}")
        logger.debug(f"Exception details: {traceback.format_exc()}")