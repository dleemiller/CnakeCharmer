# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count valid parenthesizations of length 2n (Catalan numbers via DP, Cython-optimized).

Keywords: leetcode, parentheses, catalan, dynamic programming, combinatorics, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(800,))
def valid_parentheses_gen(int n):
    """Count the number of valid parenthesizations of length 2n."""
    cdef long long mod = 1000000007
    cdef int total = 2 * n
    cdef int size = (total + 1) * (n + 1)
    cdef int i, j, mid_i, mid_j, quarter_i, quarter_j
    cdef long long val, count, dp_mid, dp_quarter
    cdef int stride = n + 1

    cdef long long *dp = <long long *>malloc(size * sizeof(long long))
    if not dp:
        raise MemoryError()

    for i in range(size):
        dp[i] = 0

    dp[0 * stride + 0] = 1

    for i in range(total):
        for j in range(n + 1):
            val = dp[i * stride + j]
            if val == 0:
                continue
            if j + 1 <= n:
                dp[(i + 1) * stride + j + 1] = (dp[(i + 1) * stride + j + 1] + val) % mod
            if j > 0:
                dp[(i + 1) * stride + j - 1] = (dp[(i + 1) * stride + j - 1] + val) % mod

    count = dp[total * stride + 0]

    cdef long long dp_diag_sum = 0
    cdef long long mid_val
    for i in range(1, total + 1):
        j = i // 2
        if j <= n:
            dp_diag_sum = (dp_diag_sum + dp[i * stride + j]) % mod

    cdef int parity_j = n % 2
    mid_val = dp[n * stride + parity_j]

    free(dp)

    return (count, dp_diag_sum, mid_val)
