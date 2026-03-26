# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Longest common substring between two deterministic strings using DP (Cython-optimized).

Keywords: dynamic programming, longest common substring, string matching, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1500,))
def longest_common_substring(int n):
    """Find longest common substring length using flat C array DP table."""
    cdef int i, j, w
    cdef int max_len = 0
    cdef long long diag_sum = 0
    cdef unsigned int hash_val
    cdef int *s1 = <int *>malloc(n * sizeof(int))
    cdef int *s2 = <int *>malloc(n * sizeof(int))
    cdef int *dp = <int *>malloc((n + 1) * (n + 1) * sizeof(int))

    if s1 == NULL or s2 == NULL or dp == NULL:
        if s1 != NULL:
            free(s1)
        if s2 != NULL:
            free(s2)
        if dp != NULL:
            free(dp)
        raise MemoryError("Failed to allocate arrays")

    w = n + 1

    # Build strings using hash-based generators
    for i in range(n):
        hash_val = <unsigned int>(i * <unsigned int>2654435761)
        s1[i] = (hash_val >> 4) % 3
        hash_val = <unsigned int>(i * <unsigned int>1640531527)
        s2[i] = (hash_val >> 4) % 3

    # Initialize first row and column to 0
    for i in range(n + 1):
        dp[i * w] = 0
        dp[i] = 0

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i * w + j] = dp[(i - 1) * w + (j - 1)] + 1
                if dp[i * w + j] > max_len:
                    max_len = dp[i * w + j]
            else:
                dp[i * w + j] = 0

    # Sum along main diagonal
    for i in range(1, n + 1):
        diag_sum += dp[i * w + i]

    free(s1)
    free(s2)
    free(dp)
    return (max_len, diag_sum)
