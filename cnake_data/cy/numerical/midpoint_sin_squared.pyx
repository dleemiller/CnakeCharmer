# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Midpoint rule integration of sin^2(x) over [0, pi].

Numerical integration using the midpoint (rectangle) rule, evaluating
sin^2 at the center of each subinterval.

Keywords: numerical, integration, midpoint, trigonometry, sin_squared, cython, benchmark
"""

from libc.math cimport sin, M_PI
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000000,))
def midpoint_sin_squared(int n):
    """Integrate sin^2(x) from 0 to pi using the midpoint rule with n steps."""
    cdef int i, half_n
    cdef double dx, dx2, x, s, total, total2
    cdef double integral_n, integral_half_n
    cdef double pi = M_PI

    # Full resolution: n steps
    dx = pi / n
    total = 0.0
    with nogil:
        for i in range(n):
            x = (i + 0.5) * dx
            s = sin(x)
            total += s * s
    integral_n = total * dx

    # Half resolution: n//2 steps
    half_n = n // 2
    if half_n < 1:
        half_n = 1
    dx2 = pi / half_n
    total2 = 0.0
    with nogil:
        for i in range(half_n):
            x = (i + 0.5) * dx2
            s = sin(x)
            total2 += s * s
    integral_half_n = total2 * dx2

    return (integral_n, integral_half_n)
