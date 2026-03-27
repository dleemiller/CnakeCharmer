# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Build a histogram from deterministic integer data using const typed memoryview for input.

Keywords: statistics, histogram, frequency, distribution, typed memoryview, const, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


cdef int _build_histogram(const int[:] data, int n, int num_bins):
    """Build histogram from const int[:] memoryview, return max bin count."""
    cdef int i, max_count
    cdef int *bins = <int *>malloc(num_bins * sizeof(int))
    if not bins:
        raise MemoryError()
    memset(bins, 0, num_bins * sizeof(int))

    for i in range(n):
        bins[data[i]] += 1

    max_count = 0
    for i in range(num_bins):
        if bins[i] > max_count:
            max_count = bins[i]

    free(bins)
    return max_count


@cython_benchmark(syntax="cy", args=(100000,))
def const_histogram(int n):
    """Build histogram using const int[:] memoryview for read-only input."""
    cdef int i
    cdef int num_bins = 100

    cdef int *ptr = <int *>malloc(n * sizeof(int))
    if not ptr:
        raise MemoryError()

    cdef int[::1] data = <int[:n]>ptr

    for i in range(n):
        data[i] = (i * 83 + 19) % 100

    # Pass as const memoryview to cdef function
    cdef int result = _build_histogram(data, n, num_bins)

    free(ptr)
    return result
