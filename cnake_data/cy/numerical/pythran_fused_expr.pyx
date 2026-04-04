# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
# cython: np_pythran=True
"""Fused element-wise expression a*b + c*d using Pythran backend.

Pythran fuses the entire expression into a single pass with no
temporary arrays, using C++ expression templates.

Keywords: numerical, fused expression, pythran, numpy, cython, benchmark
"""

import numpy as np
cimport numpy as cnp

from cnake_data.benchmarks import cython_benchmark

cnp.import_array()


@cython_benchmark(syntax="cy", args=(1000000,))
def pythran_fused_expr(int n):
    """Evaluate a*b + c*d element-wise and return sum.

    Pythran eliminates the 3 temporary arrays NumPy would create.

    Args:
        n: Array length.

    Returns:
        Sum of the fused expression.
    """
    rng = np.random.RandomState(42)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] a = rng.standard_normal(n)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] b = rng.standard_normal(n)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] c = rng.standard_normal(n)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] d = rng.standard_normal(n)
    cdef double result = np.sum(a * b + c * d)
    return result
