# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Count subsets that sum to a target value (Cython-optimized).

Keywords: dynamic programming, subset sum, counting, modular arithmetic, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500,))
def subset_sum_count(int n):
    """Count subsets summing to target using C arrays."""
    cdef long long MOD = 1000000007
    cdef int target = n * 3
    cdef int i, j, v
    cdef long long *dp = <long long *>malloc((target + 1) * sizeof(long long))
    cdef int *items = <int *>malloc(n * sizeof(int))
    cdef long long result

    if dp == NULL or items == NULL:
        if dp != NULL:
            free(dp)
        if items != NULL:
            free(items)
        raise MemoryError("Failed to allocate arrays")

    memset(dp, 0, (target + 1) * sizeof(long long))
    dp[0] = 1

    for i in range(n):
        items[i] = (i * 7 + 3) % 20 + 1

    for i in range(n):
        v = items[i]
        for j in range(target, v - 1, -1):
            dp[j] = (dp[j] + dp[j - v]) % MOD

    result = dp[target]
    cdef long long result_mid = dp[target // 2]
    free(dp)
    free(items)
    return (int(result), int(result_mid))
