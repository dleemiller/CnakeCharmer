# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Newton's method root finding for multiple starting points (Cython-optimized).

Keywords: newton, root finding, cubic, optimization, cython, benchmark
"""

from libc.math cimport fabs
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def newton_method_nd(int n):
    """Find roots of f(x) = x^3 - 2x - 5 using Newton's method."""
    cdef int count = 0
    cdef int i, j, converged
    cdef double x, fx, fpx

    for i in range(n):
        x = i * 0.01 - 2.5
        converged = 0

        for j in range(50):
            fx = x * x * x - 2.0 * x - 5.0
            fpx = 3.0 * x * x - 2.0
            if fabs(fpx) < 1e-30:
                break
            x = x - fx / fpx
            if fabs(fx) < 1e-10:
                converged = 1
                break

        count += converged

    return count
