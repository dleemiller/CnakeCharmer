# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Evaluate a polynomial at n Chebyshev nodes (Cython-optimized).

Keywords: chebyshev, polynomial, numerical, interpolation, cython, benchmark
"""

from libc.math cimport cos, sin, M_PI
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def chebyshev_nodes(int n):
    """Evaluate polynomial at Chebyshev nodes using libc.math and typed loops."""
    cdef double coeffs[10]
    cdef int k, j
    cdef double x, val, total

    for k in range(10):
        coeffs[k] = sin(k * 0.1)

    total = 0.0
    for k in range(n):
        x = cos((2 * k + 1) / (2.0 * n) * M_PI)

        # Horner's method
        val = coeffs[9]
        for j in range(8, -1, -1):
            val = val * x + coeffs[j]

        total += val

    return total
