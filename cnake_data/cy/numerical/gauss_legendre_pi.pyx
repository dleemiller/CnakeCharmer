# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute pi using the Gauss-Legendre algorithm (Cython-optimized).

Keywords: numerical, pi, gauss-legendre, iterative, convergence, cython, benchmark
"""

from libc.math cimport sqrt
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100,))
def gauss_legendre_pi(int n):
    """Compute pi using n iterations of the Gauss-Legendre algorithm."""
    cdef double a, b, t, p, a_next, result
    cdef int i

    a = 1.0
    b = 1.0 / sqrt(2.0)
    t = 0.25
    p = 1.0

    for i in range(n):
        a_next = (a + b) / 2.0
        b = sqrt(a * b)
        t = t - p * (a - a_next) * (a - a_next)
        p = 2.0 * p
        a = a_next

    result = (a + b) * (a + b) / (4.0 * t)
    return result
