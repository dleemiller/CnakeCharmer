# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Babylonian method for computing square roots (Cython-optimized).

Keywords: sqrt, babylonian, newton, numerical, iterative, convergence, cython
"""

from cnake_data.benchmarks import cython_benchmark
from libc.math cimport fabs


@cython_benchmark(syntax="cy", args=(1000000,))
def babylonian_sqrt_sum(int n):
    """Compute sqrt via Babylonian method for integers 1..n and return their sum."""
    cdef double total = 0.0
    cdef double x, x_prev, val_d
    cdef int val, j

    for val in range(1, n + 1):
        val_d = <double>val
        x = 1.0
        for j in range(100):
            x_prev = x
            x = (x + val_d / x) / 2.0
            if fabs(x_prev - x) < 1e-14:
                break
        total += x
    return total
