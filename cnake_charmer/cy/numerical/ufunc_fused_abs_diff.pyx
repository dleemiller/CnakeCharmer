# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Absolute element-wise difference using a fused-type Cython ufunc.

Defines a real_numeric fused type (int/float, no complex) so the ufunc
works for integer and float types while remaining orderable.

Keywords: numerical, absolute difference, ufunc, fused type, cython, benchmark
"""

import numpy as np
cimport cython

from cnake_charmer.benchmarks import cython_benchmark

ctypedef fused real_numeric:
    int
    long
    float
    double


@cython.ufunc
cdef real_numeric abs_diff_scalar(real_numeric a, real_numeric b) noexcept nogil:
    """Return |a - b| (works for int/long/float/double)."""
    if a > b:
        return a - b
    else:
        return b - a


@cython_benchmark(syntax="cy", args=(1000000,))
def ufunc_fused_abs_diff(int n):
    """Compute sum of |a[i] - b[i]| for two random float64 arrays."""
    rng = np.random.RandomState(42)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    result = abs_diff_scalar(a, b)
    return float(np.sum(result))
