# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute midpoint Riemann sum of f(x) = sin(x)/x (sinc) from 0.001 to 10*pi (Cython-optimized).

Keywords: numerical integration, Riemann sum, midpoint, sinc function, quadrature, cython
"""

from libc.math cimport sin, M_PI
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(10000000,))
def riemann_sum_midpoint(int n):
    """Compute midpoint Riemann sum of sin(x)/x from 0.001 to 10*pi."""
    cdef int i
    cdef double a, b, dx, total, x

    a = 0.001
    b = 10.0 * M_PI
    dx = (b - a) / n
    total = 0.0

    for i in range(n):
        x = a + (i + 0.5) * dx
        total += sin(x) / x

    return total * dx
