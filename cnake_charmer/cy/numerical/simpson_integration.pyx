# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Simpson's rule integration of f(x) = sin(x) * exp(-x/100) from 0 to 10 (Cython-optimized).

Keywords: numerical, integration, Simpson's rule, trigonometry, cython, benchmark
"""

from libc.math cimport sin, exp
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(3000000,))
def simpson_integration(int n):
    """Integrate f(x) = sin(x) * exp(-x/100) from 0 to 10 using Simpson's rule."""
    cdef int i
    cdef double a, b, h, x, result

    if n % 2 == 1:
        n += 1

    a = 0.0
    b = 10.0
    h = (b - a) / n

    result = sin(a) * exp(-a / 100.0) + sin(b) * exp(-b / 100.0)

    for i in range(1, n, 2):
        x = a + i * h
        result += 4.0 * sin(x) * exp(-x / 100.0)

    for i in range(2, n, 2):
        x = a + i * h
        result += 2.0 * sin(x) * exp(-x / 100.0)

    return result * h / 3.0
