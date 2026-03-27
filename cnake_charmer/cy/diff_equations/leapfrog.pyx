# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Leapfrog integration for a harmonic oscillator (Cython-optimized).

Integrates x'' = -x using the leapfrog (Stormer-Verlet) method.
Returns trajectory metrics and energy drift.

Keywords: ODE, leapfrog, Verlet, harmonic oscillator, symplectic, integration, cython, benchmark
"""

from libc.math cimport fabs
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000000,))
def leapfrog(int n):
    """Integrate harmonic oscillator x'' = -x using leapfrog method."""
    cdef double dt = 50.0 / n
    cdef double x = 1.0
    cdef double v = 0.0
    cdef double e0, e_final, energy_drift
    cdef int i

    # Initial energy: E = 0.5*v^2 + 0.5*x^2
    e0 = 0.5 * v * v + 0.5 * x * x

    # Half-step kick for velocity
    v = v - 0.5 * dt * x

    for i in range(n):
        # Drift (position update)
        x = x + dt * v
        # Kick (velocity update) - force = -x
        if i < n - 1:
            v = v - dt * x
        else:
            # Final half-step
            v = v - 0.5 * dt * x

    # Final energy
    e_final = 0.5 * v * v + 0.5 * x * x
    energy_drift = fabs(e_final - e0)

    return (x, v, energy_drift)
