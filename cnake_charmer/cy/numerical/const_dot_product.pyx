# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Compute the dot product of two vectors using const typed memoryviews (read-only guarantee).

Keywords: numerical, dot product, linear algebra, typed memoryview, const, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


cdef double _dot(const double[:] a, const double[:] b, int n):
    """Compute dot product from const memoryviews (read-only access)."""
    cdef int i
    cdef double result = 0.0
    for i in range(n):
        result += a[i] * b[i]
    return result


@cython_benchmark(syntax="cy", args=(100000,))
def const_dot_product(int n):
    """Compute dot product using const double[:] memoryviews."""
    cdef int i

    cdef double *ptr_a = <double *>malloc(n * sizeof(double))
    cdef double *ptr_b = <double *>malloc(n * sizeof(double))
    if not ptr_a or not ptr_b:
        if ptr_a: free(ptr_a)
        if ptr_b: free(ptr_b)
        raise MemoryError()

    cdef double[::1] a = <double[:n]>ptr_a
    cdef double[::1] b = <double[:n]>ptr_b

    for i in range(n):
        a[i] = ((i * 31 + 7) % 500) / 25.0
        b[i] = ((i * 53 + 11) % 500) / 25.0

    # Pass as const memoryviews to cdef function
    cdef double result = _dot(a, b, n)

    free(ptr_a)
    free(ptr_b)
    return result
