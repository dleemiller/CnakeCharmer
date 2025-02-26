import re
import os
import subprocess
import tempfile
import traceback

import dspy
from dspy.signatures.signature import ensure_signature
from dspy.primitives import Module
import os
from dotenv import load_dotenv

load_dotenv()

lm=dspy.LM(model="openrouter/google/gemini-2.0-flash-001")
dspy.configure(lm=lm)
class CodeGenerator(Module):
    def __init__(self, signature, max_iters=2):
        super().__init__()
        self.signature = ensure_signature(signature)
        self.max_iters = max_iters

        # Create chain for initial code generation
        self.code_generate = dspy.ChainOfThought(
            dspy.Signature(
                {
                    # The signature includes just the input prompt
                    "prompt": self.signature.fields["prompt"]
                },
                instructions=(
                    "You are given `prompt` describing a user request. "
                    "Generate code in either Python or Cython that solves the request. "
                    "Output ONLY the code in triple backticks. No extra commentary.\n"
                    "If the user specifically requests Cython, produce valid Cython code. "
                    "If it's Python, produce valid Python code.\n"
                    "Example:\n"
                    "```\n"
                    "# your code here\n"
                    "```\n"
                ),
            ),
        )

        # Create chain for regeneration if the first code fails
        self.code_regenerate = dspy.ChainOfThought(
            dspy.Signature(
                {
                    "prompt": self.signature.fields["prompt"],
                    "previous_code": dspy.InputField(
                        prefix="Previous Code:",
                        desc="Previously generated code that errored",
                        format=str
                    ),
                    "error": dspy.InputField(
                        prefix="Error:",
                        desc="Error message from compilation or runtime",
                        format=str
                    ),
                    "generated_code": self.signature.fields["generated_code"],
                },
                instructions=(
                    "You generated code previously that failed to run/compile. "
                    "The user prompt is `prompt`. The failing code is `previous_code`. "
                    "The error message is `error`.\n"
                    "Your job: correct the code and provide a working version in triple backticks, "
                    "with no additional commentary.\n"
                ),
            )
        )

    def forward(self, **kwargs):
        """
        Steps:
          1) Generate code from LLM
          2) Parse code block
          3) Attempt to compile/run
          4) If error => re-generate up to max_iters
        """
        # 1) initial generation
        code_data = self.code_generate(**kwargs)
        code_block, parse_error = self._extract_code(code_data.get("generated_code", ""))

        if parse_error:
            # If we can't parse, we can do a fallback regeneration with a parse error
            return self._try_regeneration(kwargs, "", parse_error)

        # 2) compile/run check
        error = self._compile_or_run(code_block)
        if error:
            return self._try_regeneration(kwargs, code_block, error)

        # If we reach here, no errors
        return {"generated_code": code_block, "error": None}

    def _try_regeneration(self, kwargs, previous_code, error):
        # Attempt to fix code up to self.max_iters times
        attempts = 0
        print(f"attempts: {attempts}")
        while attempts < self.max_iters:
            attempts += 1
            regen_data = self.code_regenerate(
                prompt=kwargs["prompt"],
                previous_code=previous_code,
                error=error,
            )
            new_code, parse_error = self._extract_code(regen_data.get("generated_code", ""))
            if parse_error:
                error = parse_error
                previous_code = new_code
                continue

            # Try compiling/running again
            err = self._compile_or_run(new_code)
            if not err:
                # success
                return {"generated_code": new_code, "error": None}
            else:
                error = err
                previous_code = new_code

        # If still errors after max_iters
        return {"generated_code": previous_code, "error": error}

    def _extract_code(self, text):
        """
        Extract triple-backtick code block from LLM response.
        Return (code, error_if_any).
        """
        match = re.search(r"```[\w\s]*\n?(.*?)```", text, re.DOTALL)
        if not match:
            # fallback: maybe the model didn't fence properly
            # we can just return the text as is
            code_block = text.strip()
            if not code_block:
                return ("", "ERROR: Could not parse code block.")
            return (code_block, None)
        code_block = match.group(1).strip()
        if not code_block:
            return ("", "ERROR: Empty code block after triple backticks.")
        return (code_block, None)

    def _is_cython(self, code_str):
        """
        Quick heuristic: if there's 'cdef' or if the user explicitly said 'pyx',
        we assume it's Cython code.
        """
        if "cdef" in code_str or ".pyx" in code_str or "cython" in code_str.lower():
            return True
        return False

    def _compile_or_run(self, code):
        """
        If it's recognized as Cython code, attempt to compile.
        Otherwise, attempt to run as plain Python.
        Return error string or None if success.
        """
        if self._is_cython(code):
            return self._compile_cython(code)
        else:
            return self._run_python(code)

    def _run_python(self, code_str):
        import runpy
        import textwrap
        # We'll put the code in a small temp file and run it
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            tmp_path = f.name
            f.write(code_str)
        try:
            runpy.run_path(tmp_path, run_name="__main__")
        except Exception as e:
            return f"Python run error: {traceback.format_exc()}"
        finally:
            os.remove(tmp_path)
        return None  # success

    def _compile_cython(self, code_str):
        """
        Minimal approach:
          1) write code to .pyx
          2) create a minimal setup.py
          3) run 'python setup.py build_ext --inplace'
        Return error message or None.
        """
        import textwrap, sys
        with tempfile.TemporaryDirectory() as tmpdir:
            pyx_path = os.path.join(tmpdir, "gen_code.pyx")
            with open(pyx_path, "w") as f:
                f.write(code_str)

            setup_path = os.path.join(tmpdir, "setup.py")
            with open(setup_path, "w") as f:
                f.write(textwrap.dedent(f"""
                    from setuptools import setup
                    from Cython.Build import cythonize

                    setup(
                        name="gen_code",
                        ext_modules=cythonize("{pyx_path}", language_level=3),
                    )
                """))

            # Use sys.executable, not bare "python"
            cmd = [sys.executable, setup_path, "build_ext", "--inplace"]
            try:
                proc = subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True)
                if proc.returncode != 0:
                    return f"Cython compile error:\n{proc.stderr}"
            except Exception as e:
                return f"Cython compile error: {traceback.format_exc()}"

        return None

import dspy
from dspy.signatures.signature import Signature
from dspy.signatures import InputField, OutputField

# A single, generic signature: user prompt in, code out
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


generator = CodeGenerator(signature=code_signature, max_iters=3)
result = generator.forward(prompt="Write an efficient cython method for calculating a dot product of 2 vectors.")
print("FINAL RESULT:", result)
print("Generated code:")
print(result["generated_code"])
