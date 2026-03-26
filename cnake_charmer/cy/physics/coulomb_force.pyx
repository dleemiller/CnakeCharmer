# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute total electrostatic potential energy of charges in 1D (Cython-optimized).

Keywords: physics, coulomb, electrostatic, potential, pairwise, cython, benchmark
"""

from libc.math cimport fabs
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000,))
def coulomb_force(int n):
    """Compute Coulomb potential energy with O(n^2) typed loop."""
    cdef double k = 8.9875e9
    cdef double total = 0.0
    cdef double xi, xj, qi, qj, r
    cdef int i, j

    for i in range(n):
        xi = i * 0.1
        qi = 1.0 if i % 2 == 0 else -1.0
        for j in range(i + 1, n):
            xj = j * 0.1
            qj = 1.0 if j % 2 == 0 else -1.0
            r = fabs(xi - xj)
            if r > 0:
                total += k * qi * qj / r

    return total
