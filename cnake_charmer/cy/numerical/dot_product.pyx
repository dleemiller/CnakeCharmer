# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Dot product of two vectors (Cython-optimized).

Keywords: dot product, vector, numerical, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark
import cython


@cython_benchmark(syntax="cy", args=(100000,))
def dot_product(int n):
    """Compute the dot product of two vectors using C-typed loop."""
    cdef int i
    cdef double result = 0.0
    cdef list a = [i * 0.5 for i in range(n)]
    cdef list b = [float(n - i) for i in range(n)]

    for i in range(n):
        result += <double>a[i] * <double>b[i]

    return result
