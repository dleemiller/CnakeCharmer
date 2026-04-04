# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Matrix chain multiplication with split tracking and DP diagnostics (Cython-optimized).

Keywords: dynamic programming, matrix chain, optimization, split, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200,))
def matrix_chain_order(int n):
    """Find minimum cost matrix chain multiplication with split point tracking."""
    cdef int i, j, k, length, mid, split_first
    cdef long long cost, min_cost, dp_mid
    cdef long long sentinel = 4611686018427387903  # 2**62 - 1

    if n < 2:
        return (0, 0, 0)

    cdef int *d = <int *>malloc((n + 1) * sizeof(int))
    cdef long long *dp = <long long *>malloc(n * n * sizeof(long long))
    cdef int *split = <int *>malloc(n * n * sizeof(int))

    if not d or not dp or not split:
        if d: free(d)
        if dp: free(dp)
        if split: free(split)
        raise MemoryError()

    for i in range(n + 1):
        d[i] = 15 + (i * 31 + 17) % 85

    for i in range(n * n):
        dp[i] = 0
        split[i] = 0

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i * n + j] = sentinel
            for k in range(i, j):
                cost = dp[i * n + k] + dp[(k + 1) * n + j] + <long long>d[i] * <long long>d[k + 1] * <long long>d[j + 1]
                if cost < dp[i * n + j]:
                    dp[i * n + j] = cost
                    split[i * n + j] = k

    min_cost = dp[0 * n + (n - 1)]
    split_first = split[0 * n + (n - 1)]

    mid = n // 2
    if mid > 0 and mid < n:
        dp_mid = dp[0 * n + mid]
    else:
        dp_mid = 0

    free(d)
    free(dp)
    free(split)

    return (min_cost, split_first, dp_mid)
