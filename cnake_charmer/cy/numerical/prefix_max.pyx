# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Compute prefix maximum of a deterministically generated sequence (Cython-optimized).

Keywords: numerical, prefix maximum, running max, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def prefix_max(int n):
    """Compute the prefix maximum using C-typed operations and C array."""
    cdef int i
    cdef int value
    cdef int current_max = 0
    cdef int *arr = <int *>malloc(n * sizeof(int))

    if arr == NULL:
        raise MemoryError("Failed to allocate array")

    for i in range(n):
        value = (i * 31 + 17) % 10000
        if value > current_max:
            current_max = value
        arr[i] = current_max

    cdef list result = [arr[i] for i in range(n)]

    free(arr)
    return result
