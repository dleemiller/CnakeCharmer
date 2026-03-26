# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Secant method root finding (Cython-optimized).

Keywords: numerical, root finding, secant method, iterative, cython, benchmark
"""

from libc.math cimport fabs
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def secant_method(int n):
    """Find roots of f(x) = x^3 - 2x - 5 using the secant method."""
    cdef double total = 0.0
    cdef double x0, x1, x_new, f0, f1
    cdef int i, j

    for i in range(n):
        x0 = i * 0.1
        x1 = x0 + 1.0

        f0 = x0 * x0 * x0 - 2.0 * x0 - 5.0
        f1 = x1 * x1 * x1 - 2.0 * x1 - 5.0

        for j in range(50):
            if fabs(f1 - f0) < 1e-15:
                break
            x_new = x1 - f1 * (x1 - x0) / (f1 - f0)
            x0 = x1
            f0 = f1
            x1 = x_new
            f1 = x1 * x1 * x1 - 2.0 * x1 - 5.0
            if fabs(f1) < 1e-12:
                break

        total += x1

    return total
