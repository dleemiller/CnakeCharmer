# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Maximum revenue from cutting a rod of length n (Cython-optimized).

Keywords: dynamic programming, rod cutting, optimization, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(3000,))
def rod_cutting(int n):
    """Compute maximum revenue using C arrays."""
    cdef int i, j, val, best
    cdef int *price = <int *>malloc((n + 1) * sizeof(int))
    cdef int *dp = <int *>malloc((n + 1) * sizeof(int))
    cdef int result

    if price == NULL or dp == NULL:
        if price != NULL:
            free(price)
        if dp != NULL:
            free(dp)
        raise MemoryError("Failed to allocate arrays")

    price[0] = 0
    for i in range(1, n + 1):
        price[i] = (i * 7 + 3) % 50 + 1

    memset(dp, 0, (n + 1) * sizeof(int))

    for j in range(1, n + 1):
        best = 0
        for i in range(1, j + 1):
            val = price[i] + dp[j - i]
            if val > best:
                best = val
        dp[j] = best

    result = dp[n]
    free(price)
    free(dp)
    return result
