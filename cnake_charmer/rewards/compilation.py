"""
Compilation reward: binary gate.

Returns 1.0 if the Cython code compiles, 0.0 otherwise.
This acts as a gate — other rewards are only meaningful if code compiles.
"""

from cnake_charmer.validate.compiler import cleanup_build, compile_cython


def compilation_reward(cython_code: str, **kwargs) -> float:
    """Return 1.0 if code compiles, 0.0 otherwise."""
    result = compile_cython(cython_code, annotate=False)
    cleanup_build(result)
    return 1.0 if result.success else 0.0
