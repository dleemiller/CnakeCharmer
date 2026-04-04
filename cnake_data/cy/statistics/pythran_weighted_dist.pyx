# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
# cython: np_pythran=True
"""Weighted Euclidean distance using Pythran backend.

Pythran fuses the reduction chain sqrt(sum(w * (a-b)**2)) into a
single pass per row pair with no temporary arrays.

Keywords: statistics, weighted distance, euclidean, pythran, numpy, cython, benchmark
"""

import numpy as np
cimport numpy as cnp

from cnake_data.benchmarks import cython_benchmark

cnp.import_array()


@cython_benchmark(syntax="cy", args=(500000,))
def pythran_weighted_dist(int n):
    """Compute weighted Euclidean distances between row pairs and return sum.

    Args:
        n: Total element count (reshaped as n//500 rows of 500 cols).

    Returns:
        Sum of all weighted distances.
    """
    rng = np.random.RandomState(42)
    cdef int cols = 500
    cdef int rows = n // cols
    if rows < 2:
        return 0.0

    cdef cnp.ndarray[cnp.float64_t, ndim=2] mat = rng.standard_normal((rows, cols))
    cdef cnp.ndarray[cnp.float64_t, ndim=1] w = np.abs(rng.standard_normal(cols))
    w = w / np.sum(w)

    cdef double total = 0.0
    cdef int i
    cdef cnp.ndarray[cnp.float64_t, ndim=1] diff

    for i in range(rows - 1):
        diff = mat[i] - mat[i + 1]
        total += float(np.sqrt(np.sum(w * diff * diff)))

    return total
