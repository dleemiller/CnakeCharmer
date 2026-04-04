# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count the number of ways to make change for amount n (Cython-optimized).

Keywords: dynamic programming, coin change, counting, combinatorics, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000,))
def coin_change_count(int n):
    """Count distinct ways to make change for amount n using coins [1, 2, 5, 10, 25].

    Args:
        n: Target amount.

    Returns:
        Tuple of (dp[n], dp[n//2], dp[n//4], dp[n//3]) all mod 10^9+7.
    """
    cdef long long *dp = <long long *>malloc((n + 1) * sizeof(long long))
    if not dp:
        raise MemoryError()

    cdef long long MOD = 1000000007
    cdef int i, coin
    cdef int coins[5]
    cdef long long r0, r1, r2, r3

    coins[0] = 1
    coins[1] = 2
    coins[2] = 5
    coins[3] = 10
    coins[4] = 25

    for i in range(n + 1):
        dp[i] = 0
    dp[0] = 1

    for coin in range(5):
        with nogil:
            for i in range(coins[coin], n + 1):
                dp[i] = (dp[i] + dp[i - coins[coin]]) % MOD

    r0 = dp[n]
    r1 = dp[n // 2]
    r2 = dp[n // 4]
    r3 = dp[n // 3]
    free(dp)
    return (r0, r1, r2, r3)
