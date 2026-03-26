# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute autocorrelation of a signal for lags 0..99 (Cython-optimized).

Returns the sum of autocorrelation values across all computed lags.

Keywords: dsp, autocorrelation, correlation, lag, signal, cython, benchmark
"""

from libc.math cimport sin, cos
from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def autocorrelation(int n):
    """Compute autocorrelation for lags 0..99 and return sum."""
    cdef int i, lag
    cdef int max_lag = 100
    cdef double acc, total
    cdef double *s = <double *>malloc(n * sizeof(double))
    if not s:
        raise MemoryError()

    for i in range(n):
        s[i] = sin(i * 0.1) * cos(i * 0.03)

    total = 0.0
    for lag in range(max_lag):
        acc = 0.0
        for i in range(n - lag):
            acc += s[i] * s[i + lag]
        total += acc

    free(s)
    return total
