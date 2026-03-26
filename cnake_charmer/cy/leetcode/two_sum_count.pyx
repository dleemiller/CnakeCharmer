# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count pairs in a deterministic array that sum to a target value.

Keywords: leetcode, two sum, hash map, counting, pairs, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200000,))
def two_sum_count(int n):
    """Count pairs (i, j) with i < j where arr[i] + arr[j] == target."""
    cdef int target = n // 2
    cdef int *arr = <int *>malloc(n * sizeof(int))
    cdef int *counts = <int *>malloc(n * sizeof(int))
    if not arr or not counts:
        if arr: free(arr)
        if counts: free(counts)
        raise MemoryError()

    cdef int i, val, complement
    cdef long long total = 0

    # Generate array
    for i in range(n):
        arr[i] = (i * 31 + 17) % n

    # Initialize counts to zero
    for i in range(n):
        counts[i] = 0

    # Count pairs using hash-like approach with direct indexing
    # Since values are in [0, n), we can use direct array indexing
    for i in range(n):
        val = arr[i]
        complement = target - val
        if complement >= 0 and complement < n:
            total += counts[complement]
        counts[val] += 1

    free(arr)
    free(counts)
    return int(total)
