# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Weighted bin counting with typed memoryview (Cython).

Explicit loop avoids NumPy bincount overhead for small bin counts.

Keywords: statistics, bincount, weighted, numpy, typed memoryview, cython, benchmark
"""

import numpy as np
cimport numpy as cnp

from libc.stdlib cimport calloc, free
from cnake_data.benchmarks import cython_benchmark

cnp.import_array()


@cython_benchmark(syntax="cy", args=(500000,))
def numpy_bincount_weighted(int n):
    """Weighted bin count and return max bin value."""
    rng = np.random.RandomState(42)
    cdef cnp.ndarray[int, ndim=1] idx_arr = (
        rng.randint(0, 1000, size=n).astype(np.intc)
    )
    cdef int[::1] indices = idx_arr
    cdef cnp.ndarray[double, ndim=1] w_arr = rng.random(n)
    cdef double[::1] weights = w_arr

    cdef int num_bins = 1000
    cdef double *bins = <double *>calloc(
        num_bins, sizeof(double)
    )
    if not bins:
        raise MemoryError()

    cdef int i
    cdef double max_val

    with nogil:
        for i in range(n):
            bins[indices[i]] += weights[i]

        max_val = bins[0]
        for i in range(1, num_bins):
            if bins[i] > max_val:
                max_val = bins[i]

    free(bins)
    return max_val
