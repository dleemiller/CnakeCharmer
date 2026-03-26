# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count ways to break a string into words of length 1-5 (Cython-optimized).

Keywords: word break, dynamic programming, string, counting, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def word_break(int n):
    """Count word break ways using C array for DP table."""
    cdef long long MOD = 1000000007
    cdef long long *dp = <long long *>malloc((n + 1) * sizeof(long long))
    if not dp:
        raise MemoryError()

    cdef int i, length
    cdef long long result

    dp[0] = 1
    for i in range(1, n + 1):
        dp[i] = 0
        for length in range(1, 6):
            if i - length >= 0:
                dp[i] = (dp[i] + dp[i - length]) % MOD

    result = dp[n]
    free(dp)
    return result
