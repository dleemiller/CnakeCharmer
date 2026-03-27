# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Compute cumulative sum of a deterministically generated sequence (Cython-optimized).

Keywords: cumulative sum, prefix sum, numerical, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def cumulative_sum(int n):
    """Compute the cumulative sum using C-typed accumulation and C array."""
    cdef long long total = 0
    cdef int i
    cdef long long *arr = <long long *>malloc(n * sizeof(long long))

    if arr == NULL:
        raise MemoryError("Failed to allocate array")

    for i in range(n):
        total += (i * 13 + 7) % 1000
        arr[i] = total

    cdef list result = [arr[i] for i in range(n)]

    free(arr)
    return result
