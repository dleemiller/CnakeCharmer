# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Count unique paths in a grid with obstacles using DP (Cython-optimized).

Keywords: dynamic programming, grid paths, obstacles, combinatorics, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(800,))
def count_paths_grid(int n):
    """Count unique paths in an n x n grid with obstacles using flat C array."""
    cdef long long MOD = 1000000007
    cdef int i, j, mid
    cdef long long *dp = <long long *>malloc(n * n * sizeof(long long))
    cdef char *blocked = <char *>malloc(n * n * sizeof(char))
    cdef long long result, result_mid

    if dp == NULL or blocked == NULL:
        if dp != NULL:
            free(dp)
        if blocked != NULL:
            free(blocked)
        raise MemoryError("Failed to allocate arrays")

    # Build obstacle grid
    for i in range(n):
        for j in range(n):
            if (i * 7 + j * 13 + 3) % 17 == 0:
                blocked[i * n + j] = 1
            else:
                blocked[i * n + j] = 0

    # Ensure start and end are clear
    blocked[0] = 0
    blocked[(n - 1) * n + (n - 1)] = 0

    # Initialize all dp to 0
    for i in range(n * n):
        dp[i] = 0

    dp[0] = 1

    # First column
    for i in range(1, n):
        if blocked[i * n] == 0:
            dp[i * n] = dp[(i - 1) * n]

    # First row
    for j in range(1, n):
        if blocked[j] == 0:
            dp[j] = dp[j - 1]

    # Fill table
    for i in range(1, n):
        for j in range(1, n):
            if blocked[i * n + j] == 0:
                dp[i * n + j] = (dp[(i - 1) * n + j] + dp[i * n + (j - 1)]) % MOD

    mid = n / 2
    result = dp[(n - 1) * n + (n - 1)]
    result_mid = dp[mid * n + mid]

    free(dp)
    free(blocked)
    return (result, result_mid)
