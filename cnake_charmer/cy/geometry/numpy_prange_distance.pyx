# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Parallel pairwise distance with prange (Cython).

Uses prange over outer loop for parallel pairwise Euclidean
distance computation on 3D points.

Keywords: geometry, distance, pairwise, prange, numpy,
    typed memoryview, cython, benchmark
"""

import numpy as np
cimport numpy as cnp

from cython.parallel import prange
from libc.math cimport sqrt
from cnake_charmer.benchmarks import cython_benchmark

cnp.import_array()


@cython_benchmark(syntax="cy", args=(2000,))
def numpy_prange_distance(int n):
    """Sum all pairwise distances for n 3D points."""
    rng = np.random.RandomState(42)
    cdef cnp.ndarray[double, ndim=2] pts_arr = (
        rng.standard_normal((n, 3)).astype(np.float64)
    )
    cdef double[:, ::1] pts = pts_arr
    cdef cnp.ndarray[double, ndim=1] row_sums_arr = (
        np.zeros(n, dtype=np.float64)
    )
    cdef double[::1] row_sums = row_sums_arr
    cdef int i, j
    cdef double dx, dy, dz, d, total

    for i in prange(n, nogil=True):
        for j in range(i + 1, n):
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            dz = pts[i, 2] - pts[j, 2]
            d = sqrt(dx * dx + dy * dy + dz * dz)
            row_sums[i] += d

    total = 0.0
    with nogil:
        for i in range(n):
            total += row_sums[i]

    return total
