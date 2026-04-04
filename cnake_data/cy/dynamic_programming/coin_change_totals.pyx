# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Minimum coin change for amounts 1..n with total tracking (Cython-optimized).

Keywords: dynamic programming, coin change, minimum coins, optimization, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200000,))
def coin_change_totals(int n):
    """Compute minimum coins for each amount from 1 to n."""
    cdef int *dp = <int *>malloc((n + 1) * sizeof(int))
    cdef int coins[5]
    cdef int i, j, c
    cdef long long total_coins = 0
    cdef int mid = n // 2

    if not dp:
        raise MemoryError()

    coins[0] = 1
    coins[1] = 3
    coins[2] = 7
    coins[3] = 11
    coins[4] = 23

    dp[0] = 0
    for i in range(1, n + 1):
        dp[i] = dp[i - 1] + 1  # coin of value 1
        for j in range(1, 5):
            c = coins[j]
            if c <= i and dp[i - c] + 1 < dp[i]:
                dp[i] = dp[i - c] + 1

    for i in range(1, n + 1):
        total_coins += dp[i]

    cdef int coins_for_n = dp[n]
    cdef int coins_for_mid = dp[mid]

    free(dp)
    return (coins_for_n, coins_for_mid, total_coins)
