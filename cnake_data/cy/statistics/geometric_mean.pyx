# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Geometric mean via log-sum-exp trick to avoid overflow (Cython-optimized).

Keywords: statistics, geometric mean, log, overflow, numerical, cython, benchmark
"""

from libc.math cimport log, exp, sin, fabs
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(3000000,))
def geometric_mean(int n):
    """Compute geometric mean of positive deterministic values using log-sum."""
    cdef int i
    cdef double log_sum = 0.0, min_val = 1e308
    cdef double val, log_mean, gmean

    for i in range(n):
        val = 1.0 + ((i * 31 + 11) % 200) / 10.0 + 0.5 * fabs(sin(i * 0.03))
        log_sum += log(val)
        if val < min_val:
            min_val = val

    log_mean = log_sum / n
    gmean = exp(log_mean)

    return (gmean, log_sum, min_val)
