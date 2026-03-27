# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Parallel row-wise L2 norm with prange (Cython).

Uses prange over rows with typed 2D memoryview for parallel
norm computation.

Keywords: numerical, norm, L2, rows, prange, numpy, typed memoryview, cython, benchmark
"""

import numpy as np
cimport numpy as cnp

from cython.parallel import prange
from libc.math cimport sqrt
from cnake_charmer.benchmarks import cython_benchmark

cnp.import_array()


@cython_benchmark(syntax="cy", args=(2000,))
def numpy_prange_norm(int n):
    """Compute row-wise L2 norms of n x 1000 matrix."""
    rng = np.random.RandomState(42)
    cdef int cols = 1000
    cdef cnp.ndarray[double, ndim=2] mat_arr = (
        rng.standard_normal((n, cols)).astype(np.float64)
    )
    cdef double[:, ::1] mat = mat_arr
    cdef cnp.ndarray[double, ndim=1] norms_arr = (
        np.empty(n, dtype=np.float64)
    )
    cdef double[::1] norms = norms_arr
    cdef int i, j
    cdef double total = 0.0

    for i in prange(n, nogil=True):
        norms[i] = 0.0
        for j in range(cols):
            norms[i] = norms[i] + mat[i, j] * mat[i, j]
        norms[i] = sqrt(norms[i])

    with nogil:
        for i in range(n):
            total += norms[i]

    return total
