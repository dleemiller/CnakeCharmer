# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Minimum coin change (Cython-optimized with C array).

Keywords: dynamic programming, coin change, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(50000,))
def coin_change(int n):
    """Find minimum coins to make amount n using C-level DP table."""
    cdef int *dp = <int *>malloc((n + 1) * sizeof(int))
    if not dp:
        raise MemoryError()

    cdef int i, val, result, result_mid
    cdef int coins[4]
    coins[0] = 1
    coins[1] = 5
    coins[2] = 10
    coins[3] = 25

    dp[0] = 0

    for i in range(1, n + 1):
        dp[i] = dp[i - 1] + 1  # Using coin of value 1
        # Check coin 5
        if 5 <= i:
            val = dp[i - 5] + 1
            if val < dp[i]:
                dp[i] = val
        # Check coin 10
        if 10 <= i:
            val = dp[i - 10] + 1
            if val < dp[i]:
                dp[i] = val
        # Check coin 25
        if 25 <= i:
            val = dp[i - 25] + 1
            if val < dp[i]:
                dp[i] = val

    result = dp[n]
    result_mid = dp[n // 2]
    free(dp)
    return (result, result_mid)
