# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Cumulative maximum with typed memoryview (Cython).

Uses typed memoryview for fast sequential cumulative max.

Keywords: numerical, cumulative, maximum, numpy, typed memoryview, cython, benchmark
"""

import numpy as np
cimport numpy as cnp

from cnake_data.benchmarks import cython_benchmark

cnp.import_array()


@cython_benchmark(syntax="cy", args=(500000,))
def numpy_cummax(int n):
    """Compute cumulative max and return sum of result."""
    rng = np.random.RandomState(42)
    cdef cnp.ndarray[double, ndim=1] data_arr = (
        rng.standard_normal(n).astype(np.float64)
    )
    cdef double[::1] data = data_arr
    cdef cnp.ndarray[double, ndim=1] out_arr = (
        np.empty(n, dtype=np.float64)
    )
    cdef double[::1] out = out_arr
    cdef double current_max
    cdef double total = 0.0
    cdef int i

    with nogil:
        current_max = data[0]
        out[0] = current_max
        for i in range(1, n):
            if data[i] > current_max:
                current_max = data[i]
            out[i] = current_max

        for i in range(n):
            total += out[i]

    return total
