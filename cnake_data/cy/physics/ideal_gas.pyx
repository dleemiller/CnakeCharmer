# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute pressure for ideal gas states using PV=nRT (Cython-optimized).

Keywords: physics, ideal, gas, pressure, thermodynamics, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000000,))
def ideal_gas(int n):
    """Compute ideal gas pressures with pure typed loop."""
    cdef double R = 8.314
    cdef double n_mol = 1.0
    cdef double total = 0.0
    cdef double V, T, P
    cdef int i

    for i in range(n):
        V = (i % 100 + 1) * 0.001
        T = (i % 500 + 200)
        P = n_mol * R * T / V
        total += P

    return total
