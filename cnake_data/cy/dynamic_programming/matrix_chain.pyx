# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Minimum scalar multiplications for matrix chain multiplication (Cython-optimized).

Keywords: dynamic programming, matrix chain, optimization, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200,))
def matrix_chain(int n):
    """Find minimum scalar multiplications using flat C arrays for DP table."""
    cdef int i, j, k, length
    cdef long long cost
    cdef long long *dp = <long long *>malloc(n * n * sizeof(long long))
    cdef int *d = <int *>malloc((n + 1) * sizeof(int))
    cdef long long result

    if dp == NULL or d == NULL:
        if dp != NULL:
            free(dp)
        if d != NULL:
            free(d)
        raise MemoryError("Failed to allocate arrays")

    memset(dp, 0, n * n * sizeof(long long))

    for i in range(n + 1):
        d[i] = 10 + (i * 7 + 3) % 90

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i * n + j] = 4611686018427387903  # 2^62 - 1
            for k in range(i, j):
                cost = dp[i * n + k] + dp[(k + 1) * n + j] + <long long>d[i] * <long long>d[k + 1] * <long long>d[j + 1]
                if cost < dp[i * n + j]:
                    dp[i * n + j] = cost

    result = dp[0 * n + (n - 1)]
    free(dp)
    free(d)
    return result
