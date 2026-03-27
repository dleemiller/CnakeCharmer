# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Histogram with cnp.float64_t typed processing (Cython).

Manual binning with typed memoryview and cnp types.

Keywords: statistics, histogram, numpy, cnp, typed memoryview, cython, benchmark
"""

import numpy as np
cimport numpy as cnp

from libc.stdlib cimport calloc, free
from cnake_charmer.benchmarks import cython_benchmark

cnp.import_array()


@cython_benchmark(syntax="cy", args=(500000,))
def numpy_typed_histogram(int n):
    """Compute histogram with 256 bins, return max count."""
    rng = np.random.RandomState(42)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] data_arr = rng.standard_normal(n)
    cdef cnp.float64_t[::1] data = data_arr

    cdef int num_bins = 256
    cdef cnp.float64_t lo = -4.0
    cdef cnp.float64_t hi = 4.0
    cdef cnp.float64_t bin_width = (hi - lo) / num_bins

    cdef int *counts = <int *>calloc(
        num_bins, sizeof(int)
    )
    if not counts:
        raise MemoryError()

    cdef int i, b
    cdef cnp.float64_t val
    cdef int max_count
    cdef cnp.float64_t inv_bin_width = 1.0 / bin_width

    with nogil:
        for i in range(n):
            val = data[i]
            if val < lo or val >= hi:
                continue
            b = <int>((val - lo) * inv_bin_width)
            if b >= num_bins:
                b = num_bins - 1
            counts[b] += 1

        max_count = counts[0]
        for b in range(1, num_bins):
            if counts[b] > max_count:
                max_count = counts[b]

    free(counts)
    return max_count
