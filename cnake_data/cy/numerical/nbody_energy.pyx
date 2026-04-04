# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute total gravitational potential energy of n bodies (Cython-optimized).

Keywords: numerical, n-body, gravitational, physics, pairwise, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, sin, cos
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2000,))
def nbody_energy(int n):
    """Compute total gravitational potential energy using C arrays and libc math."""
    cdef double *xs = <double *>malloc(n * sizeof(double))
    cdef double *ys = <double *>malloc(n * sizeof(double))
    cdef double *zs = <double *>malloc(n * sizeof(double))
    cdef double *masses = <double *>malloc(n * sizeof(double))

    if not xs or not ys or not zs or not masses:
        if xs: free(xs)
        if ys: free(ys)
        if zs: free(zs)
        if masses: free(masses)
        raise MemoryError()

    cdef int i, j
    cdef double G = 6.674e-11
    cdef double total = 0.0
    cdef double dx, dy, dz, r

    for i in range(n):
        xs[i] = sin(i * 0.1)
        ys[i] = cos(i * 0.2)
        zs[i] = sin(i * 0.3)
        masses[i] = 1.0 + (i % 5) * 0.5

    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dz = zs[i] - zs[j]
            r = sqrt(dx * dx + dy * dy + dz * dz)
            if r > 0:
                total += -G * masses[i] * masses[j] / r

    free(xs)
    free(ys)
    free(zs)
    free(masses)
    return total
