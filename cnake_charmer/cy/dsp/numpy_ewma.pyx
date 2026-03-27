# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Exponentially weighted moving average (Cython with memoryview).

Uses typed memoryview over NumPy array for fast sequential
EWMA computation with nogil.

Keywords: dsp, ewma, moving average, numpy, typed memoryview, cython, benchmark
"""

import numpy as np
cimport numpy as cnp

from cnake_charmer.benchmarks import cython_benchmark

cnp.import_array()


@cython_benchmark(syntax="cy", args=(500000,))
def numpy_ewma(int n):
    """Compute EWMA and return the last value."""
    rng = np.random.RandomState(42)
    cdef cnp.ndarray[double, ndim=1] data = (
        rng.standard_normal(n).astype(np.float64)
    )
    cdef double[::1] view = data
    cdef double alpha = 0.01
    cdef double one_minus_alpha = 1.0 - alpha
    cdef double result = 0.0
    cdef int i

    with nogil:
        for i in range(n):
            result = alpha * view[i] + one_minus_alpha * result

    return result
