# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Dot product of two vectors (Cython-optimized with C arrays).

Keywords: dot product, vector, numerical, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def dot_product(int n):
    """Compute the dot product of two vectors using C-typed arrays."""
    cdef double *a = <double *>malloc(n * sizeof(double))
    cdef double *b = <double *>malloc(n * sizeof(double))
    if not a or not b:
        if a: free(a)
        if b: free(b)
        raise MemoryError()

    cdef int i
    cdef double result = 0.0

    for i in range(n):
        a[i] = i * 0.5
        b[i] = <double>(n - i)

    for i in range(n):
        result += a[i] * b[i]

    free(a)
    free(b)
    return result
