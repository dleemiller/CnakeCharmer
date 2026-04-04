# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sigmoid activation using a fused-type Cython ufunc.

Uses cython.floating so the ufunc works for both float32 and float64 arrays.

Keywords: nn_ops, sigmoid, activation, ufunc, fused type, cython, benchmark
"""

import numpy as np
cimport cython
from libc.math cimport exp, expf

from cnake_data.benchmarks import cython_benchmark


@cython.ufunc
cdef cython.floating fused_sigmoid_scalar(cython.floating x) nogil:
    """Sigmoid: 1 / (1 + exp(-x)), works for float32 and float64."""
    if cython.floating is float:
        return 1.0 / (1.0 + expf(-x))
    else:
        return 1.0 / (1.0 + exp(-x))


@cython_benchmark(syntax="cy", args=(1000000,))
def ufunc_fused_sigmoid(int n):
    """Apply sigmoid ufunc to n standard-normal float64 values and return sum."""
    rng = np.random.RandomState(42)
    arr = rng.standard_normal(n)
    result = fused_sigmoid_scalar(arr)
    return float(np.sum(result))
