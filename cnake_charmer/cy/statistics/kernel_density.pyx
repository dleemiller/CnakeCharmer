# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Gaussian kernel density estimation (Cython-optimized).

Keywords: kernel density, KDE, Gaussian, statistics, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport exp, sqrt, M_PI
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(50000,))
def kernel_density(int n):
    """Gaussian KDE at 100 evaluation points using C arrays."""
    cdef double h = 0.5
    cdef double inv_h = 1.0 / h
    cdef double norm_factor = 1.0 / (n * h * sqrt(2.0 * M_PI))
    cdef int n_eval = 100

    cdef double *data = <double *>malloc(n * sizeof(double))
    if not data:
        raise MemoryError()

    cdef int i, ei
    cdef double x, density, u, total

    for i in range(n):
        data[i] = ((i * 17 + 5) % 1000) / 100.0

    total = 0.0
    for ei in range(n_eval):
        x = ei * 10.0 / (n_eval - 1)
        density = 0.0
        for i in range(n):
            u = (x - data[i]) * inv_h
            density += exp(-0.5 * u * u)
        total += density * norm_factor

    free(data)
    return total
