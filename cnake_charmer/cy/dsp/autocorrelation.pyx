# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute autocorrelation of a deterministic signal for lags 0..99 (Cython-optimized).

Returns discriminating tuple of autocorrelation values.

Keywords: dsp, autocorrelation, correlation, lag, signal, cython, benchmark
"""

from libc.math cimport sin, cos
from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def autocorrelation(int n):
    """Compute autocorrelation for lags 0..99 and return key values."""
    cdef int i, lag
    cdef int max_lag = 100
    cdef double acc
    cdef double *s = <double *>malloc(n * sizeof(double))
    cdef double *r = <double *>malloc(max_lag * sizeof(double))
    if not s or not r:
        raise MemoryError()

    for i in range(n):
        s[i] = sin(i * 0.1) * cos(i * 0.03)

    for lag in range(max_lag):
        acc = 0.0
        for i in range(n - lag):
            acc += s[i] * s[i + lag]
        r[lag] = acc

    cdef double r0 = r[0]
    cdef double r_mid = r[max_lag // 2]
    cdef double r_last = r[max_lag - 1]

    free(s)
    free(r)
    return (r0, r_mid, r_last)
