# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Deterministic quasi-Monte Carlo integration using Halton sequences (Cython-optimized).

Keywords: simulation, Monte Carlo, quasi-random, Halton sequence, integration, cython, benchmark
"""

from libc.math cimport sin, cos, exp, sqrt, M_PI
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(300000,))
def monte_carlo_integration(int n):
    """Estimate multi-dimensional integrals using Halton quasi-random sequences."""
    cdef int sphere_count = 0
    cdef double sincos_sum = 0.0
    cdef double sqrt_sum = 0.0
    cdef double pi = M_PI
    cdef int i, idx
    cdef double x, y, z, f, r2

    for i in range(1, n + 1):
        # Halton base 2
        x = 0.0
        f = 0.5
        idx = i
        while idx > 0:
            x += f * (idx % 2)
            idx = idx // 2
            f *= 0.5

        # Halton base 3
        y = 0.0
        f = 1.0 / 3.0
        idx = i
        while idx > 0:
            y += f * (idx % 3)
            idx = idx // 3
            f /= 3.0

        # Halton base 5
        z = 0.0
        f = 0.2
        idx = i
        while idx > 0:
            z += f * (idx % 5)
            idx = idx // 5
            f *= 0.2

        # Sphere octant test
        r2 = x * x + y * y + z * z
        if r2 <= 1.0:
            sphere_count += 1

        # sin*cos*exp integral
        sincos_sum += sin(pi * x) * cos(pi * y) * exp(-z)

        # sqrt integral
        sqrt_sum += sqrt(x * y * z + 1e-30)

    cdef double sphere_estimate = <double>sphere_count / n
    cdef double sincos_estimate = sincos_sum / n
    cdef double sqrt_estimate = sqrt_sum / n

    return (sphere_estimate, sincos_estimate, sqrt_estimate)
