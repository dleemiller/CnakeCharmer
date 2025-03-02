"""
Builder implementations for ephemeral code execution.

This package provides builders for compiling and executing code in ephemeral environments.
"""

from ephemeral_runner.builders.base import BaseBuilder
from ephemeral_runner.builders.python import PythonBuilder
from ephemeral_runner.builders.cython import CythonBuilder

__all__ = ["BaseBuilder", "PythonBuilder", "CythonBuilder"]


def get_builder(code_str: str, request_id: str = None) -> BaseBuilder:
    """
    Factory function to get the appropriate builder for the given code.

    Args:
        code_str: Code to build
        request_id: Unique identifier for this build request

    Returns:
        An instance of the appropriate builder
    """
    # Create a temporary base builder to use the is_cython method
    temp_builder = BaseBuilder(request_id)

    if temp_builder.is_cython(code_str):
        return CythonBuilder(request_id)
    else:
        return PythonBuilder(request_id)
