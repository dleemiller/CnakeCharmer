# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Clear every other block using memset.

Keywords: algorithms, memset, clear, pattern, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def memset_clear_pattern(int n):
    """Zero every other 32-element block with memset."""
    cdef int *arr = <int *>malloc(n * sizeof(int))
    if not arr:
        raise MemoryError()

    cdef int i, b, start, num_blocks
    cdef long long total = 0
    cdef int block = 32

    for i in range(n):
        arr[i] = (
            (<long long>i * <long long>2654435761 + 17)
            ^ (<long long>i * <long long>1664525)
        ) & 0xFFFF

    num_blocks = n // block
    for b in range(0, num_blocks, 2):
        start = b * block
        memset(&arr[start], 0, block * sizeof(int))

    for i in range(n):
        total += arr[i]

    free(arr)
    return total
