# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute Planck spectral radiance at multiple wavelengths (Cython-optimized).

Keywords: physics, blackbody, planck, radiation, spectral, cython, benchmark
"""

from libc.math cimport exp
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def blackbody_radiation(int n):
    """Compute Planck spectral radiance with typed loop and libc exp."""
    cdef double h = 6.626e-34
    cdef double c = 2.998e8
    cdef double kb = 1.381e-23
    cdef double T = 5778.0
    cdef double two_hc2 = 2.0 * h * c * c
    cdef double hc_over_kT = h * c / (kb * T)
    cdef double total = 0.0
    cdef double lam, lam2, lam5, exponent, B
    cdef int i

    for i in range(n):
        lam = (0.1 + i * 0.01) * 1e-6
        lam2 = lam * lam
        lam5 = lam2 * lam2 * lam
        exponent = hc_over_kT / lam
        if exponent > 700.0:
            continue
        B = two_hc2 / lam5 / (exp(exponent) - 1.0)
        total += B

    return total
