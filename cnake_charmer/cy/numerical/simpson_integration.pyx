# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Simpson's rule integration of sin(x)*exp(-x/n) over [0, n].

Keywords: numerical, integration, Simpson's rule, trigonometry, cython, benchmark
"""

from libc.math cimport sin, exp
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(3000000,))
def simpson_integration(int n):
    """Integrate f(x) = sin(x) * exp(-x/n) from 0 to n using Simpson's rule."""
    cdef int i, panels
    cdef double a, b, h, x, result, integral, mid_x, midpoint_contrib
    cdef double fn = <double>n

    panels = n
    if panels % 2 == 1:
        panels += 1

    a = 0.0
    b = fn
    h = (b - a) / panels

    result = sin(a) * exp(-a / fn) + sin(b) * exp(-b / fn)

    for i in range(1, panels, 2):
        x = a + i * h
        result += 4.0 * sin(x) * exp(-x / fn)

    for i in range(2, panels, 2):
        x = a + i * h
        result += 2.0 * sin(x) * exp(-x / fn)

    integral = result * h / 3.0

    mid_x = b / 2.0
    midpoint_contrib = sin(mid_x) * exp(-mid_x / fn)

    return (integral, midpoint_contrib, panels)
