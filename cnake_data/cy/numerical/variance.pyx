# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute population variance using two-pass algorithm (Cython-optimized with C arrays).

Keywords: numerical, variance, statistics, two-pass, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def variance(int n):
    """Compute population variance using C arrays and cdef double arithmetic."""
    cdef double *values = <double *>malloc(n * sizeof(double))
    if not values:
        raise MemoryError()

    cdef int i
    cdef double total = 0.0
    cdef double mean, diff, var_sum

    # Build values and compute sum in one pass
    for i in range(n):
        values[i] = (i * 17 + 5) % 1000 / 10.0
        total += values[i]

    mean = total / n

    # Second pass: sum of squared deviations
    var_sum = 0.0
    for i in range(n):
        diff = values[i] - mean
        var_sum += diff * diff

    free(values)
    return var_sum / n
