# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute numerical derivative using central differences (Cython-optimized).

Keywords: numerical derivative, central difference, differentiation, cython, benchmark
"""

from libc.math cimport sin, exp
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def numerical_derivative(int n):
    """Compute numerical derivative of f(x)=sin(x)*exp(-x/100) at n points."""
    if n < 3:
        return 0.0

    cdef double dx = 10.0 / (n - 1)
    cdef double h = dx
    cdef double inv_2h = 1.0 / (2.0 * h)
    cdef double total = 0.0
    cdef int i
    cdef double x_plus, x_minus, f_plus, f_minus

    for i in range(1, n - 1):
        x_plus = (i + 1) * dx
        x_minus = (i - 1) * dx
        f_plus = sin(x_plus) * exp(-x_plus / 100.0)
        f_minus = sin(x_minus) * exp(-x_minus / 100.0)
        total += (f_plus - f_minus) * inv_2h

    return total
