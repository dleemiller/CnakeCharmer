# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Evaluate a polynomial at multiple points using Horner's method (Cython-optimized).

Keywords: numerical, polynomial, evaluation, horner, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000,))
def polynomial_eval(int n):
    """Evaluate degree-n polynomial at n points using Horner's method with C arrays."""
    cdef double *coeffs = <double *>malloc(n * sizeof(double))
    cdef double *points = <double *>malloc(n * sizeof(double))

    if not coeffs or not points:
        if coeffs: free(coeffs)
        if points: free(points)
        raise MemoryError()

    cdef int i, k
    cdef double total = 0.0
    cdef double result, x

    for i in range(n):
        coeffs[i] = (i * 7 + 3) % 100 / 10.0

    for i in range(n):
        points[i] = (i * 13 + 7) % 1000 / 1000.0

    for i in range(n):
        x = points[i]
        result = coeffs[n - 1]
        for k in range(n - 2, -1, -1):
            result = result * x + coeffs[k]
        total += result

    free(coeffs)
    free(points)
    return total
