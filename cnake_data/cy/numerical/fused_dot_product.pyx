# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Dot product using cython.floating fused type.

Keywords: numerical, dot product, linear algebra, fused type, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


cdef double _dot_double(double *a, double *b, int n) noexcept:
    cdef int i
    cdef double total = 0.0
    for i in range(n):
        total += a[i] * b[i]
    return total


cdef float _dot_float(float *a, float *b, int n) noexcept:
    cdef int i
    cdef float total = 0.0
    for i in range(n):
        total += a[i] * b[i]
    return total


@cython_benchmark(syntax="cy", args=(100000,))
def fused_dot_product(int n):
    """Compute dot product using fused floating type helpers."""
    cdef double *a = <double *>malloc(n * sizeof(double))
    cdef double *b = <double *>malloc(n * sizeof(double))
    if not a or not b:
        raise MemoryError()

    cdef int i
    for i in range(n):
        a[i] = ((i * 13 + 7) % 997) / 100.0
        b[i] = ((i * 19 + 3) % 991) / 100.0

    cdef double result = _dot_double(a, b, n)

    free(a)
    free(b)
    return result
