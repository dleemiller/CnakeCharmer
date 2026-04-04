# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Length of longest palindromic subsequence using DP (Cython-optimized).

Keywords: dynamic programming, palindrome, subsequence, string, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(3000,))
def longest_palindrome(int n):
    """Find the length of the longest palindromic subsequence using C arrays."""
    cdef int *s = <int *>malloc(n * sizeof(int))
    cdef int *prev = <int *>malloc(n * sizeof(int))
    cdef int *curr = <int *>malloc(n * sizeof(int))
    cdef int *tmp

    if s == NULL or prev == NULL or curr == NULL:
        if s != NULL:
            free(s)
        if prev != NULL:
            free(prev)
        if curr != NULL:
            free(curr)
        raise MemoryError("Failed to allocate arrays")

    cdef int i, j, a, b

    # Generate string codes
    for i in range(n):
        s[i] = (i * 7 + 3) % 26

    # Initialize rows to 0
    for i in range(n):
        prev[i] = 0
        curr[i] = 0

    # dp[j] = LPS for s[i..j], iterate i from n-1 down to 0
    for i in range(n - 1, -1, -1):
        curr[i] = 1  # single char base case
        for j in range(i + 1, n):
            if s[i] == s[j]:
                curr[j] = prev[j - 1] + 2
            else:
                a = prev[j]
                b = curr[j - 1]
                curr[j] = a if a > b else b
        # Swap rows
        tmp = prev
        prev = curr
        curr = tmp

    cdef int result = prev[n - 1]
    cdef int result_mid = prev[n // 2]
    free(s)
    free(prev)
    free(curr)
    return (result, result_mid)
