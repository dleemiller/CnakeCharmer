"""
Core functionality for ephemeral code execution.

This module provides the main entry point for the ephemeral_runner package,
which allows for the generation, compilation, and execution of code in
ephemeral environments.
"""

import os
import re
import logging
import uuid
from typing import Optional, Dict, Any, Tuple

import dspy
from dspy.primitives import Module
from dspy.signatures import InputField, OutputField
from dspy.signatures.signature import Signature, ensure_signature

from cnake_charmer.generate.ephemeral_runner.builders import get_builder
from cnake_charmer.generate.ephemeral_runner.exceptions import ParseError

# Configure logger with more detailed formatting
logger = logging.getLogger("ephemeral_runner")
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s"
)
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


class EphemeralCodeGenerator(Module):
    """
    Main class for generating and executing code in ephemeral environments.

    This class:
    1. Generates code using an LLM
    2. Parses the generated code
    3. Creates ephemeral environments
    4. Compiles and runs the code
    5. Handles errors and regenerates code if needed
    """

    def __init__(self, signature, max_iters=3, lm=None):
        """
        Initialize the code generator.

        Args:
            signature: DSPy signature for the generation task
            max_iters: Maximum number of regeneration iterations
            lm: Language model to use for generation
        """
        super().__init__()
        self.signature = ensure_signature(signature)
        self.max_iters = max_iters
        self.lm = lm
        logger.info(f"Initialized EphemeralCodeGenerator with max_iters={max_iters}")

        # Chain for initial generation
        self.code_generate = dspy.ChainOfThought(
            Signature(
                {
                    "prompt": self.signature.fields["prompt"],
                    "generated_code": self.signature.fields["generated_code"],
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
                ),
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
                        format=str,
                    ),
                    "error": InputField(
                        prefix="Error:",
                        desc="Error message from compilation or runtime",
                        format=str,
                    ),
                    "generated_code": self.signature.fields["generated_code"],
                },
                instructions=(
                    "You generated code previously that failed to run/compile. "
                    "The user prompt is `prompt`. The failing code is `previous_code`. "
                    "The error message is `error`.\n"
                    "Your job: correct the code and provide a working version in triple backticks, "
                    "with no extra commentary.\n\n"
                    "Make sure to include ALL necessary imports and cimports.\n"
                    "Make sure all required libraries are properly imported."
                ),
            )
        )
        logger.debug("Initialized code_regenerate chain")

    def forward(self, **kwargs):
        """
        Generate, build, and run code.

        Args:
            **kwargs: Keyword arguments for the generation task

        Returns:
            Dictionary with generated code and error message if any
        """
        request_id = str(uuid.uuid4())
        logger.info(
            f"Forward called with prompt [ID: {request_id}]: {kwargs.get('prompt', '')[:100]}..."
        )

        # Step 1: get initial code
        try:
            logger.debug(f"Request {request_id}: Calling code_generate")
            code_data = self.code_generate(**kwargs)
            logger.debug(
                f"Request {request_id}: code_generate returned {type(code_data)}"
            )

            raw_code = ""
            if hasattr(code_data, "generated_code"):
                raw_code = code_data.generated_code
                logger.debug(
                    f"Request {request_id}: Extracted code from Prediction object"
                )
            else:
                raw_code = code_data.get("generated_code", "")
                logger.debug(f"Request {request_id}: Extracted code from dictionary")

            logger.debug(
                f"Request {request_id}: Initial generation raw output: {raw_code[:200]}..."
            )
        except Exception as e:
            logger.error(
                f"Request {request_id}: Initial code generation failed: {str(e)}"
            )
            return {"generated_code": "", "error": f"Code generation failed: {str(e)}"}

        # Step 2: parse
        code_block, parse_err = self._extract_code(raw_code, request_id)
        if parse_err:
            logger.warning(
                f"Request {request_id}: Parse error => regeneration: {parse_err}"
            )
            return self._try_regeneration(
                kwargs, previous_code="", error=parse_err, request_id=request_id
            )

        # Step 3: build and run
        builder = get_builder(code_block, request_id)
        error = builder.build_and_run(code_block)
        if error:
            logger.warning(
                f"Request {request_id}: Build error => regeneration: {error[:1000]}..."
            )
            return self._try_regeneration(
                kwargs, previous_code=code_block, error=error, request_id=request_id
            )

        logger.info(f"Request {request_id}: Successfully generated and built code")
        return {"generated_code": code_block, "error": None}

    def _try_regeneration(self, kwargs, previous_code, error, request_id=None):
        """
        Try to regenerate code after a failure.

        Args:
            kwargs: Original generation arguments
            previous_code: Code that failed
            error: Error message
            request_id: Request ID for logging

        Returns:
            Dictionary with generated code and error message if any
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        attempts = 0
        while attempts < self.max_iters:
            attempts += 1
            logger.info(
                f"Request {request_id}: Attempting regeneration, attempt #{attempts}/{self.max_iters}"
            )

            # Log the inputs to regeneration for debugging
            logger.debug(
                f"Request {request_id}: Regeneration input prompt: {kwargs.get('prompt', '')[:50]}..."
            )
            logger.debug(
                f"Request {request_id}: Regeneration previous code length: {len(previous_code)}"
            )
            logger.debug(
                f"Request {request_id}: Regeneration error: {error[:100]}..."
                if len(error) > 100
                else error
            )

            try:
                regen_data = self.code_regenerate(
                    prompt=kwargs["prompt"], previous_code=previous_code, error=error
                )

                # Handle Prediction objects from DSPy
                logger.debug(
                    f"Request {request_id}: Regeneration returned type: {type(regen_data)}"
                )

                new_raw = ""
                if hasattr(regen_data, "generated_code"):
                    new_raw = regen_data.generated_code
                    logger.debug(f"Request {request_id}: Used generated_code attribute")
                else:
                    new_raw = regen_data.get("generated_code", "")
                    logger.debug(f"Request {request_id}: Used dictionary access")

                logger.debug(
                    f"Request {request_id}: Regenerated code length: {len(new_raw)}"
                )

            except Exception as e:
                logger.error(
                    f"Request {request_id}: Regeneration attempt #{attempts} failed: {str(e)}"
                )
                continue

            new_code, parse_err = self._extract_code(new_raw, request_id)
            if parse_err:
                # next iteration
                logger.warning(
                    f"Request {request_id}: Parse error on regenerated code (attempt #{attempts}) => continuing: {parse_err}"
                )
                previous_code = new_raw
                error = parse_err
                continue

            builder = get_builder(new_code, request_id)
            build_err = builder.build_and_run(new_code)
            if build_err:
                logger.warning(
                    f"Request {request_id}: Build error again (attempt #{attempts}) => continuing: {build_err[:300]}..."
                )
                error = build_err
                previous_code = new_code
            else:
                # success
                logger.info(
                    f"Request {request_id}: Regeneration successful on attempt #{attempts}"
                )
                return {"generated_code": new_code, "error": None}

        # if we exhaust attempts
        logger.error(
            f"Request {request_id}: Exhausted all {self.max_iters} regeneration attempts, still has error"
        )
        return {"generated_code": previous_code, "error": error}

    def _extract_code(self, text, request_id=None):
        """
        Extract code from text, looking for triple backtick code blocks.

        Args:
            text: Text to extract code from
            request_id: Request ID for logging

        Returns:
            Tuple of (code, error_message)
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        logger.debug(
            f"Request {request_id}: Extracting code from text of length {len(text)}"
        )

        # Handle empty or None text
        if not text:
            logger.error(f"Request {request_id}: Empty text input to code extraction")
            return ("", "ERROR: Empty code text input.")

        try:
            match = re.search(r"```[\w\s]*\n?(.*?)```", text, re.DOTALL)
            if not match:
                logger.debug(
                    f"Request {request_id}: No triple backticks found, first 100 chars: {text[:100]}..."
                )
                code_block = text.strip()
                if not code_block:
                    logger.error(
                        f"Request {request_id}: Could not parse code block - empty content"
                    )
                    return ("", "ERROR: Could not parse code block.")
                logger.warning(
                    f"Request {request_id}: No triple backticks found, using entire text as code"
                )
                return (code_block, None)

            code_block = match.group(1).strip()
            if not code_block:
                logger.error(
                    f"Request {request_id}: Empty code block after triple backticks"
                )
                return ("", "ERROR: Empty code block after triple backticks.")

            # Check if we need to handle multiple code blocks
            all_code_blocks = re.findall(r"```[\w\s]*\n?(.*?)```", text, re.DOTALL)
            if len(all_code_blocks) > 1:
                logger.info(
                    f"Request {request_id}: Found {len(all_code_blocks)} code blocks, using the first one"
                )

            logger.info(
                f"Request {request_id}: Successfully extracted code block ({len(code_block)} characters)"
            )

            logger.debug(
                f"Request {request_id}: Code block begins with: {code_block[:100]}..."
                if len(code_block) > 100
                else code_block
            )
            return (code_block, None)
        except Exception as e:
            logger.error(
                f"Request {request_id}: Exception during code extraction: {str(e)}"
            )
            return ("", f"ERROR: Code extraction failed: {str(e)}")


def generate_code(prompt, max_iters=3, lm=None):
    """
    Generate, build, and run code from a prompt.

    Args:
        prompt: Description of the code to generate
        max_iters: Maximum number of regeneration iterations
        lm: Language model to use for generation

    Returns:
        Dictionary with generated code and error message if any
    """
    code_signature = Signature(
        {
            "prompt": InputField(
                prefix="User Prompt:",
                desc="The user request describing what code to generate",
                format=str,
            ),
            "generated_code": OutputField(
                prefix="Code:",
                desc="The code snippet that solves the user request",
                format=str,
            ),
        }
    )

    generator = EphemeralCodeGenerator(
        signature=code_signature, max_iters=max_iters, lm=lm
    )
    return generator.forward(prompt=prompt)
