# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute left Riemann sum of f(x) = x^2 * exp(-x) from 0 to 10 (Cython-optimized).

Keywords: numerical integration, Riemann sum, left endpoint, quadrature, cython
"""

from libc.math cimport exp
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(10000000,))
def riemann_sum_left(int n):
    """Compute left Riemann sum of x^2 * exp(-x) from 0 to 10."""
    cdef int i
    cdef double dx, total, x

    dx = 10.0 / n
    total = 0.0

    for i in range(n):
        x = i * dx
        total += x * x * exp(-x)

    return total * dx
