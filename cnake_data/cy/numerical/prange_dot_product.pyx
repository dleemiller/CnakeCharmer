# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Parallel dot product using prange with += reduction.

Keywords: numerical, dot product, reduction, prange, parallel, cython, benchmark
"""

from cython.parallel import prange
from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def prange_dot_product(int n):
    """Dot product with prange += reduction."""
    cdef double *a = <double *>malloc(n * sizeof(double))
    cdef double *b = <double *>malloc(n * sizeof(double))
    if not a or not b:
        free(a)
        free(b)
        raise MemoryError()

    cdef int i
    cdef unsigned int h
    for i in range(n):
        h = <unsigned int>(
            <long long>i * <long long>2654435761
        ) & <unsigned int>0xFFFFFFFF
        a[i] = h / 4294967296.0
        h = <unsigned int>(
            <long long>i * <long long>2246822519
        ) & <unsigned int>0xFFFFFFFF
        b[i] = h / 4294967296.0

    cdef double total = 0.0
    for i in prange(n, nogil=True):
        total += a[i] * b[i]

    free(a)
    free(b)
    return total
