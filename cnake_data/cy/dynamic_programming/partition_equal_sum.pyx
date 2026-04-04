# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Check if an array can be partitioned into two subsets with equal sum (Cython-optimized).

Keywords: dynamic programming, partition, subset sum, bitset, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2000,))
def partition_equal_sum(int n):
    """Check if array can be partitioned into two equal-sum subsets using C array DP."""
    cdef int *items = <int *>malloc(n * sizeof(int))
    if items == NULL:
        raise MemoryError("Failed to allocate items")

    cdef int i, j, total, target, val, reachable_count

    total = 0
    for i in range(n):
        items[i] = (i * 7 + 3) % 20 + 1
        total += items[i]

    if total % 2 != 0:
        free(items)
        return (0, 0)

    target = total // 2

    # Boolean DP array: dp[j] = 1 if sum j is reachable
    cdef char *dp = <char *>malloc((target + 1) * sizeof(char))
    if dp == NULL:
        free(items)
        raise MemoryError("Failed to allocate dp")

    memset(dp, 0, (target + 1) * sizeof(char))
    dp[0] = 1

    for i in range(n):
        val = items[i]
        for j in range(target, val - 1, -1):
            if dp[j - val]:
                dp[j] = 1

    cdef int result = dp[target]
    reachable_count = 0
    for j in range(target + 1):
        if dp[j]:
            reachable_count += 1
    free(dp)
    free(items)
    return (1 if result else 0, reachable_count)
