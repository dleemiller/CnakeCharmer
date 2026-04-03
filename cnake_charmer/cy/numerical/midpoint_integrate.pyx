# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Midpoint-rule numerical integration of multiple transcendental functions.

Keywords: numerical integration, midpoint rule, sin, exp, transcendental, cython
"""

from libc.math cimport sin, exp

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(0.0, 10.0, 200000))
def midpoint_integrate(double start, double stop, int n):
    """Integrate sin(x)^2, exp(-x^2), and x*sin(x) over [start, stop]."""
    cdef double dx = (stop - start) / n
    cdef double sum_sin_sq = 0.0
    cdef double sum_gauss = 0.0
    cdef double sum_xsinx = 0.0
    cdef double x, s
    cdef int i

    for i in range(n):
        x = start + (i + 0.5) * dx
        s = sin(x)
        sum_sin_sq += s * s
        sum_gauss += exp(-x * x)
        sum_xsinx += x * s

    return (sum_sin_sq * dx, sum_gauss * dx, sum_xsinx * dx)
