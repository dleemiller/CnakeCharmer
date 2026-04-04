# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Build histogram using calloc for zero-initialized bins.

Keywords: statistics, histogram, calloc, cython, benchmark
"""

from libc.stdlib cimport calloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def calloc_histogram(int n):
    """Build histogram into 1024 calloc'd bins."""
    cdef int *bins = <int *>calloc(1024, sizeof(int))
    if not bins:
        raise MemoryError()

    cdef int i, val, max_count

    for i in range(n):
        val = (
            (<long long>i * <long long>2654435761 + 17)
            >> 4
        ) & 1023
        bins[val] += 1

    max_count = 0
    for i in range(1024):
        if bins[i] > max_count:
            max_count = bins[i]

    free(bins)
    return max_count
