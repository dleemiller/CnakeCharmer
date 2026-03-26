# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Minimum cost path in a grid from top-left to bottom-right (Cython-optimized).

Keywords: dynamic programming, minimum cost, grid path, optimization, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1200,))
def min_cost_path(int n):
    """Find minimum cost path in an n x n grid using flat C array."""
    cdef int i, j, mid, cost
    cdef long long *dp = <long long *>malloc(n * n * sizeof(long long))
    cdef long long result, result_mid

    if dp == NULL:
        raise MemoryError("Failed to allocate array")

    dp[0] = ((0 * 31 + 0 * 37 + 5) % 100) + 1

    # First row
    for j in range(1, n):
        cost = ((0 * 31 + j * 37 + 5) % 100) + 1
        dp[j] = dp[j - 1] + cost

    # First column
    for i in range(1, n):
        cost = ((i * 31 + 0 * 37 + 5) % 100) + 1
        dp[i * n] = dp[(i - 1) * n] + cost

    # Fill table
    for i in range(1, n):
        for j in range(1, n):
            cost = ((i * 31 + j * 37 + 5) % 100) + 1
            if dp[(i - 1) * n + j] < dp[i * n + (j - 1)]:
                dp[i * n + j] = dp[(i - 1) * n + j] + cost
            else:
                dp[i * n + j] = dp[i * n + (j - 1)] + cost

    mid = n / 2
    result = dp[(n - 1) * n + (n - 1)]
    result_mid = dp[mid * n + mid]

    free(dp)
    return (result, result_mid)
