# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Clamp array elements using a Cython ufunc.

Uses @cython.ufunc on a 3-argument cdef to clamp element-wise.

Keywords: numerical, clamp, clip, ufunc, cython, benchmark
"""

import numpy as np
cimport cython

from cnake_charmer.benchmarks import cython_benchmark


@cython.ufunc
cdef double clamp_scalar(double x, double lo, double hi) nogil:
    """Clamp x to [lo, hi]."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@cython_benchmark(syntax="cy", args=(1000000,))
def ufunc_clamp(int n):
    """Clamp n random values to [-2.5, 2.5] and return sum."""
    rng = np.random.RandomState(42)
    arr = rng.standard_normal(n)
    result = clamp_scalar(arr, -2.5, 2.5)
    return float(np.sum(result))
