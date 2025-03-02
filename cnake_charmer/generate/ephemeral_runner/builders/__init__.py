from cnake_charmer.generate.ephemeral_runner.builders.base import BaseBuilder
from cnake_charmer.generate.ephemeral_runner.builders.python import PythonBuilder
from cnake_charmer.generate.ephemeral_runner.builders.cython import CythonBuilder

__all__ = ["BaseBuilder", "PythonBuilder", "CythonBuilder"]

def get_builder(language: str, request_id: str = None) -> BaseBuilder:
    """
    Factory function to get the appropriate builder.

    Args:
        language (str): 'python' or 'cython'.
        request_id (Optional[str]): Unique build request identifier.

    Returns:
        BaseBuilder: An instance of the appropriate builder.
    """
    if language.lower() == "cython":
        return CythonBuilder(request_id)
    return PythonBuilder(request_id)