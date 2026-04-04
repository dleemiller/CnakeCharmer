# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Reverse subarrays using typed memoryview slicing and compute a checksum.

Keywords: algorithms, reverse, slicing, subarray, typed memoryview, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def memview_slice_reverse(int n):
    """Reverse blocks of 64 elements using memoryview slicing, return sum."""
    cdef int block = 64
    cdef int i, left, right
    cdef double total, tmp

    cdef double *ptr = <double *>malloc(n * sizeof(double))
    if not ptr:
        raise MemoryError()

    cdef double[::1] data = <double[:n]>ptr

    for i in range(n):
        data[i] = ((i * 43 + 17) % 1000) / 10.0

    # Reverse each block of 64 using memoryview slicing
    i = 0
    while i + block <= n:
        # Use memoryview slice for the block and reverse via swap
        left = i
        right = i + block - 1
        while left < right:
            tmp = data[left]
            data[left] = data[right]
            data[right] = tmp
            left += 1
            right -= 1
        i += block

    total = 0.0
    for i in range(n):
        total += data[i]

    free(ptr)
    return total
