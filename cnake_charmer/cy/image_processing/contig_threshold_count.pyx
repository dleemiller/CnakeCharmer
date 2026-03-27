# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Apply a threshold to 1D pixel data using a C-contiguous unsigned char memoryview.

Keywords: image processing, threshold, binary, pixel, typed memoryview, contiguous, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def contig_threshold_count(int n):
    """Apply threshold using C-contiguous unsigned char[::1] memoryview."""
    cdef int i, count
    cdef unsigned char threshold = 128

    cdef unsigned char *ptr = <unsigned char *>malloc(n * sizeof(unsigned char))
    if not ptr:
        raise MemoryError()

    cdef unsigned char[::1] pixels = <unsigned char[:n]>ptr

    for i in range(n):
        pixels[i] = <unsigned char>((i * 47 + 23) % 256)

    count = 0
    for i in range(n):
        if pixels[i] > threshold:
            count += 1

    free(ptr)
    return count
