# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Romberg integration of sin(x) (Cython-optimized).

Keywords: numerical, integration, romberg, richardson extrapolation, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin, M_PI
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(25,))
def romberg_integration(int n):
    """Compute Romberg integration of sin(x) from 0 to pi with n levels."""
    cdef double a = 0.0
    cdef double b = M_PI
    cdef int i, j, k
    cdef double h, total, factor

    # Flat array for n x n tableau
    cdef double *R = <double *>malloc(n * n * sizeof(double))
    if not R:
        raise MemoryError()

    # R[0][0] = basic trapezoidal rule
    R[0] = 0.5 * (b - a) * (sin(a) + sin(b))

    for i in range(1, n):
        h = (b - a) / (1 << i)

        # Add new midpoints
        total = 0.0
        for k in range(1, (1 << i), 2):
            total += sin(a + k * h)

        R[i * n + 0] = 0.5 * R[(i - 1) * n + 0] + h * total

        # Richardson extrapolation
        factor = 4.0
        for j in range(1, i + 1):
            R[i * n + j] = (factor * R[i * n + j - 1] - R[(i - 1) * n + j - 1]) / (factor - 1.0)
            factor *= 4.0

    cdef double result = R[(n - 1) * n + n - 1]
    free(R)
    return result
