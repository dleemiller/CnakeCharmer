# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Cross-correlation of two signals for n lags (Cython-optimized).

Computes the cross-correlation using direct O(n^2) summation and returns
the maximum value.

Keywords: dsp, cross-correlation, correlation, lag, signal, cython, benchmark
"""

from libc.math cimport sin
from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(3000,))
def cross_correlation(int n):
    """Compute cross-correlation and return the maximum value."""
    cdef int i, lag
    cdef double acc, max_corr
    cdef double *x = <double *>malloc(n * sizeof(double))
    cdef double *y = <double *>malloc(n * sizeof(double))
    if not x or not y:
        raise MemoryError()

    for i in range(n):
        x[i] = sin(i * 0.1)
        y[i] = sin(i * 0.1 + 0.5)

    max_corr = -1e300
    for lag in range(n):
        acc = 0.0
        for i in range(n - lag):
            acc += x[i] * y[i + lag]
        if acc > max_corr:
            max_corr = acc

    free(x)
    free(y)
    return max_corr
