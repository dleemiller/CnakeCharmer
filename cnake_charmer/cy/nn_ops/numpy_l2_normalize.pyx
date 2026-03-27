# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""L2-normalize rows with typed 2D memoryview (Cython).

Row-by-row normalization avoids creating large temporary arrays.

Keywords: nn_ops, L2, normalize, numpy, typed memoryview, cython, benchmark
"""

import numpy as np
cimport numpy as cnp

from libc.math cimport sqrt
from cnake_charmer.benchmarks import cython_benchmark

cnp.import_array()


@cython_benchmark(syntax="cy", args=(1000,))
def numpy_l2_normalize(int n):
    """L2-normalize rows of n x 256 matrix, return sum."""
    rng = np.random.RandomState(42)
    cdef int cols = 256
    cdef cnp.ndarray[double, ndim=2] mat_arr = (
        rng.standard_normal((n, cols)).astype(np.float64)
    )
    cdef double[:, ::1] mat = mat_arr
    cdef double norm_sq, inv_norm, total
    cdef int i, j

    total = 0.0
    with nogil:
        for i in range(n):
            norm_sq = 0.0
            for j in range(cols):
                norm_sq += mat[i, j] * mat[i, j]
            if norm_sq < 1e-24:
                norm_sq = 1e-24
            inv_norm = 1.0 / sqrt(norm_sq)
            for j in range(cols):
                mat[i, j] *= inv_norm
                total += mat[i, j]

    return total
