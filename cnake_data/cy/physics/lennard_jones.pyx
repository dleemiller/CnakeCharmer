# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute Lennard-Jones potential energy for particles in 1D (Cython-optimized).

Keywords: physics, lennard-jones, potential, molecular, pairwise, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin, fabs
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(3000,))
def lennard_jones(int n):
    """Compute Lennard-Jones potential with C arrays and O(n^2) typed loop."""
    cdef double *x = <double *>malloc(n * sizeof(double))
    if not x:
        raise MemoryError()

    cdef double epsilon = 1.0
    cdef double sigma = 1.0
    cdef double total = 0.0
    cdef double r, sr, sr6, sr12
    cdef int i, j

    for i in range(n):
        x[i] = i * 1.0 + sin(i * 0.1) * 0.1

    for i in range(n):
        for j in range(i + 1, n):
            r = fabs(x[i] - x[j])
            if r > 0:
                sr = sigma / r
                sr6 = sr * sr * sr * sr * sr * sr
                sr12 = sr6 * sr6
                total += 4.0 * epsilon * (sr12 - sr6)

    free(x)
    return total
