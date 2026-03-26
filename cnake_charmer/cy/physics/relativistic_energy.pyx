# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute relativistic kinetic energy for particles (Cython-optimized).

Keywords: physics, relativistic, energy, lorentz, gamma, cython, benchmark
"""

from libc.math cimport sqrt
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000000,))
def relativistic_energy(int n):
    """Compute relativistic kinetic energy with typed loop and libc sqrt."""
    cdef double c = 299792458.0
    cdef double c2 = c * c
    cdef double m = 1e-27
    cdef double total = 0.0
    cdef double v, v2_over_c2, gamma, KE
    cdef int i

    for i in range(n):
        v = (i * 7 + 3) % 299000000
        v2_over_c2 = (v * v) / c2
        if v2_over_c2 >= 1.0:
            continue
        gamma = 1.0 / sqrt(1.0 - v2_over_c2)
        KE = (gamma - 1.0) * m * c2
        total += KE

    return total
