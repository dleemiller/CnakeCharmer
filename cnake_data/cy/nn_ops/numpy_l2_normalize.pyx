# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""L2-normalize rows with typed 2D memoryview (Cython).

Fuses norm computation, scaling, and summation into two passes per row
(one for squared-norm, one for scale+sum) with zero temporary arrays.
The loop-carried dependency on `total` is broken by accumulating a
per-row `row_sum` and adding it once per row, enabling inner-loop
vectorization.

Keywords: nn_ops, L2, normalize, numpy, typed memoryview, cython, benchmark
"""

import numpy as np
cimport numpy as cnp

from libc.math cimport sqrt
from cnake_data.benchmarks import cython_benchmark

cnp.import_array()


@cython_benchmark(syntax="cy", args=(1000,))
def numpy_l2_normalize(int n):
    """L2-normalize rows of n x 256 matrix, return sum."""
    rng = np.random.RandomState(42)
    cdef int cols = 256
    cdef cnp.ndarray[double, ndim=2] mat_arr = rng.standard_normal((n, cols))
    cdef double[:, ::1] mat = mat_arr
    cdef double norm_sq, inv_norm, total, row_sum, val
    cdef int i, j

    total = 0.0
    with nogil:
        for i in range(n):
            # Pass 1: compute squared norm (vectorizable, no loop-carried dep)
            norm_sq = 0.0
            for j in range(cols):
                val = mat[i, j]
                norm_sq += val * val
            if norm_sq < 1e-24:
                norm_sq = 1e-24
            inv_norm = 1.0 / sqrt(norm_sq)

            # Pass 2: scale in-place and accumulate row sum separately
            # row_sum has no cross-row dependency; inner loop is vectorizable
            row_sum = 0.0
            for j in range(cols):
                val = mat[i, j] * inv_norm
                mat[i, j] = val
                row_sum += val
            total += row_sum

    return total
