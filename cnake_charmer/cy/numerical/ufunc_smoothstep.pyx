# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Hermite smoothstep using a Cython ufunc.

Hardcodes edge0=0.2, edge1=0.8 so the ufunc takes a single argument.

Keywords: numerical, smoothstep, interpolation, hermite, ufunc, cython, benchmark
"""

import numpy as np
cimport cython

from cnake_charmer.benchmarks import cython_benchmark


@cython.ufunc
cdef double smoothstep_scalar(double x) nogil:
    """Smoothstep with edge0=0.2, edge1=0.8."""
    cdef double t
    t = (x - 0.2) / (0.8 - 0.2)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return t * t * (3.0 - 2.0 * t)


@cython_benchmark(syntax="cy", args=(1000000,))
def ufunc_smoothstep(int n):
    """Apply smoothstep to n uniform-random values and return sum."""
    rng = np.random.RandomState(42)
    arr = rng.random(n)
    result = smoothstep_scalar(arr)
    return float(np.sum(result))
