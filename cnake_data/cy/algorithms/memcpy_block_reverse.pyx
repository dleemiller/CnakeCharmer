# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Reverse array in blocks of 64 using memcpy.

Keywords: algorithms, memcpy, block reverse, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def memcpy_block_reverse(int n):
    """Reverse array in 64-element blocks with memcpy."""
    cdef int *arr = <int *>malloc(n * sizeof(int))
    cdef int *tmp = <int *>malloc(64 * sizeof(int))
    if not arr or not tmp:
        if arr:
            free(arr)
        if tmp:
            free(tmp)
        raise MemoryError()

    cdef int i, lo, hi, num_blocks
    cdef long long checksum = 0
    cdef int block = 64

    for i in range(n):
        arr[i] = (
            (<long long>i * <long long>2654435761 + 17)
            ^ (i >> 3)
        ) & 0x7FFFFFFF

    num_blocks = n // block
    for i in range(num_blocks // 2):
        lo = i * block
        hi = (num_blocks - 1 - i) * block
        memcpy(tmp, &arr[lo], block * sizeof(int))
        memcpy(&arr[lo], &arr[hi], block * sizeof(int))
        memcpy(&arr[hi], tmp, block * sizeof(int))

    for i in range(0, n, 128):
        checksum += arr[i]

    free(arr)
    free(tmp)
    return checksum
