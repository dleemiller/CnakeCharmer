# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Edit distance between two deterministic strings using full DP table (Cython-optimized).

Keywords: dynamic programming, edit distance, levenshtein, dp table, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1500,))
def edit_distance_dp(int n):
    """Compute edit distance using a flat C array DP table."""
    cdef int i, j, val
    cdef int *s1 = <int *>malloc(n * sizeof(int))
    cdef int *s2 = <int *>malloc(n * sizeof(int))
    cdef int *dp = <int *>malloc((n + 1) * (n + 1) * sizeof(int))
    cdef int result, result_mid
    cdef int w = n + 1  # row width

    if s1 == NULL or s2 == NULL or dp == NULL:
        if s1 != NULL:
            free(s1)
        if s2 != NULL:
            free(s2)
        if dp != NULL:
            free(dp)
        raise MemoryError("Failed to allocate arrays")

    # Build strings as int arrays
    for i in range(n):
        s1[i] = (i * 7 + 3) % 26
        s2[i] = (i * 11 + 5) % 26

    # Initialize base cases
    for i in range(n + 1):
        dp[i * w] = i
    for j in range(n + 1):
        dp[j] = j

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i * w + j] = dp[(i - 1) * w + (j - 1)]
            else:
                val = dp[(i - 1) * w + j]
                if dp[i * w + (j - 1)] < val:
                    val = dp[i * w + (j - 1)]
                if dp[(i - 1) * w + (j - 1)] < val:
                    val = dp[(i - 1) * w + (j - 1)]
                dp[i * w + j] = 1 + val

    result = dp[n * w + n]
    result_mid = dp[(n // 2) * w + (n // 2)]
    free(s1)
    free(s2)
    free(dp)
    return (result, result_mid)
