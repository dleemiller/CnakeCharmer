# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Compute prefix sums using a C-contiguous typed memoryview and return the final element.

Keywords: numerical, prefix sum, scan, typed memoryview, contiguous, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def contig_prefix_sum(int n):
    """Compute prefix sums using C-contiguous double[::1] memoryview."""
    cdef int i
    cdef double *ptr = <double *>malloc(n * sizeof(double))
    if not ptr:
        raise MemoryError()

    cdef double[::1] data = <double[:n]>ptr

    for i in range(n):
        data[i] = ((i * 37 + 13) % 1000) / 100.0

    for i in range(1, n):
        data[i] = data[i - 1] + data[i]

    cdef double result = data[n - 1]
    free(ptr)
    return result
