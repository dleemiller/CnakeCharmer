# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute orbital periods for planets using Kepler's third law (Cython-optimized).

Keywords: physics, orbital, kepler, period, gravitational, cython, benchmark
"""

from libc.math cimport sqrt, M_PI
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000000,))
def orbital_mechanics(int n):
    """Compute orbital periods using C-typed loop and libc sqrt."""
    cdef double G = 6.674e-11
    cdef double M_sun = 1.989e30
    cdef double AU = 1.496e11
    cdef double GM = G * M_sun
    cdef double two_pi = 2.0 * M_PI
    cdef double total = 0.0
    cdef double a, a3
    cdef int i

    for i in range(n):
        a = (0.5 + i * 0.3) * AU
        a3 = a * a * a
        total += two_pi * sqrt(a3 / GM)

    return total
