# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Longest common subsequence length of two deterministic sequences (Cython-optimized).

Keywords: algorithms, LCS, longest common subsequence, dynamic programming, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2000,))
def longest_common_subsequence(int n):
    """Compute LCS length using flat C array DP table."""
    cdef int *a = <int *>malloc(n * sizeof(int))
    cdef int *b = <int *>malloc(n * sizeof(int))
    cdef int stride = n + 1
    cdef int *dp = <int *>malloc(stride * stride * sizeof(int))

    if not a or not b or not dp:
        if a: free(a)
        if b: free(b)
        if dp: free(dp)
        raise MemoryError()

    cdef int i, j, val1, val2

    # Generate sequences
    for i in range(n):
        a[i] = (i * 7 + 3) % 20
        b[i] = (i * 11 + 5) % 20

    # Initialize DP table
    memset(dp, 0, stride * stride * sizeof(int))

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i * stride + j] = dp[(i - 1) * stride + (j - 1)] + 1
            else:
                val1 = dp[(i - 1) * stride + j]
                val2 = dp[i * stride + (j - 1)]
                dp[i * stride + j] = val1 if val1 > val2 else val2

    cdef int result = dp[n * stride + n]

    free(a)
    free(b)
    free(dp)
    return result
