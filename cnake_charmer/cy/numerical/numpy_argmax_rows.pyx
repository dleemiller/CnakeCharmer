# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Row-wise argmax with typed 2D memoryview (Cython).

Explicit loop over rows and columns for argmax.

Keywords: numerical, argmax, rows, numpy, typed memoryview, cython, benchmark
"""

import numpy as np
cimport numpy as cnp

from cnake_charmer.benchmarks import cython_benchmark

cnp.import_array()


@cython_benchmark(syntax="cy", args=(1000,))
def numpy_argmax_rows(int n):
    """Find argmax of each row in n x 512 matrix."""
    rng = np.random.RandomState(42)
    cdef int cols = 512
    cdef cnp.ndarray[double, ndim=2] mat_arr = (
        rng.standard_normal((n, cols)).astype(np.float64)
    )
    cdef double[:, ::1] mat = mat_arr
    cdef long total = 0
    cdef int i, j, best_j
    cdef double best_val

    with nogil:
        for i in range(n):
            best_val = mat[i, 0]
            best_j = 0
            for j in range(1, cols):
                if mat[i, j] > best_val:
                    best_val = mat[i, j]
                    best_j = j
            total += best_j

    return total
