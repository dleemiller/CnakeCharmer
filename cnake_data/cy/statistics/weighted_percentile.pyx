# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute weighted 50th percentile of n generated values (Cython-optimized).

Keywords: weighted, percentile, median, statistics, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def weighted_percentile(int n):
    """Compute weighted 50th percentile using C arrays and counting sort."""
    cdef int *values = <int *>malloc(n * sizeof(int))
    cdef int *weights = <int *>malloc(n * sizeof(int))
    cdef int *indices = <int *>malloc(n * sizeof(int))
    if not values or not weights or not indices:
        free(values); free(weights); free(indices)
        raise MemoryError()

    cdef int i
    cdef long long total_weight, cumulative
    cdef double threshold, result

    # Generate values and weights
    for i in range(n):
        values[i] = (i * 7 + 3) % 1000
        weights[i] = (i * 13 + 7) % 10 + 1

    # Use counting sort since values are in [0, 999]
    # Count occurrences weighted by index
    cdef int max_val = 1000
    cdef long long *cum_weight = <long long *>malloc(max_val * sizeof(long long))
    if not cum_weight:
        free(values); free(weights); free(indices)
        raise MemoryError()

    for i in range(max_val):
        cum_weight[i] = 0

    total_weight = 0
    for i in range(n):
        cum_weight[values[i]] += weights[i]
        total_weight += weights[i]

    threshold = total_weight * 0.5
    cumulative = 0
    result = 0.0
    for i in range(max_val):
        cumulative += cum_weight[i]
        if cumulative >= threshold:
            result = <double>i
            break

    free(values)
    free(weights)
    free(indices)
    free(cum_weight)
    return result
