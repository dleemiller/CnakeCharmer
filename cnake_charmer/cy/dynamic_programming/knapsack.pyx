# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
0/1 Knapsack problem (Cython-optimized).

Keywords: dynamic programming, knapsack, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark
import cython
from libc.stdlib cimport malloc, free
from libc.string cimport memset


@cython_benchmark(syntax="cy", args=(500,))
def knapsack(int n):
    """Solve 0/1 knapsack using typed C arrays."""
    cdef int capacity = n * 3
    cdef int i, c, w, v, new_val
    cdef int *dp = <int *>malloc((capacity + 1) * sizeof(int))
    cdef int *weights = <int *>malloc(n * sizeof(int))
    cdef int *values = <int *>malloc(n * sizeof(int))
    cdef int result

    if dp == NULL or weights == NULL or values == NULL:
        if dp != NULL:
            free(dp)
        if weights != NULL:
            free(weights)
        if values != NULL:
            free(values)
        raise MemoryError("Failed to allocate arrays")

    memset(dp, 0, (capacity + 1) * sizeof(int))

    for i in range(n):
        weights[i] = i % 10 + 1
        values[i] = i * 3 % 17 + 1

    for i in range(n):
        w = weights[i]
        v = values[i]
        for c in range(capacity, w - 1, -1):
            new_val = dp[c - w] + v
            if new_val > dp[c]:
                dp[c] = new_val

    result = dp[capacity]
    free(dp)
    free(weights)
    free(values)
    return result
