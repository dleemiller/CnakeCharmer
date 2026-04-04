# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Classify floats into bins using a Cython ufunc.

The ufunc returns the bin index floor(x * num_bins) as a long integer.

Keywords: statistics, bins, classification, histogram, ufunc, cython, benchmark
"""

import numpy as np
cimport cython

from cnake_data.benchmarks import cython_benchmark

DEF NUM_BINS = 20


@cython.ufunc
cdef long classify_bin_scalar(double x, long num_bins) nogil:
    """Return bin index for x in [0, 1): floor(x * num_bins)."""
    cdef long idx
    idx = <long>(x * num_bins)
    if idx < 0:
        idx = 0
    elif idx >= num_bins:
        idx = num_bins - 1
    return idx


@cython_benchmark(syntax="cy", args=(1000000,))
def ufunc_classify_bin(int n):
    """Classify n random values into 20 bins and return weighted bin sum."""
    rng = np.random.RandomState(42)
    arr = rng.random(n)
    bins = classify_bin_scalar(arr, NUM_BINS)
    counts = np.bincount(bins, minlength=NUM_BINS)
    return int(np.sum(counts * np.arange(1, NUM_BINS + 1)))
