# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Parallel sum of squares using prange with += reduction.

Keywords: numerical, sum, squares, reduction, prange, parallel, cython, benchmark
"""

from cython.parallel import prange
from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def prange_sum_squares(int n):
    """Sum of squares with prange += reduction."""
    cdef double *arr = <double *>malloc(
        n * sizeof(double)
    )
    if not arr:
        raise MemoryError()

    cdef int i
    cdef unsigned int h
    for i in range(n):
        h = <unsigned int>(
            <long long>i * <long long>2654435761
        ) & <unsigned int>0xFFFFFFFF
        arr[i] = h / 4294967296.0

    cdef double total = 0.0
    for i in prange(n, nogil=True):
        total += arr[i] * arr[i]

    free(arr)
    return total
