# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Velocity Verlet integration for harmonic oscillator (Cython-optimized).

Keywords: ODE, Verlet, integration, harmonic oscillator, differential equation, numerical, cython
"""

from libc.math cimport fabs, M_PI
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(10000000,))
def verlet_integration(int n):
    """Integrate harmonic oscillator x''=-x using velocity Verlet for n steps."""
    cdef int i
    cdef double dt, x, v, a, a_new, e0, e, energy_drift_sum

    dt = 20.0 * M_PI / n
    x = 1.0
    v = 0.0

    e0 = 0.5 * (v * v + x * x)
    energy_drift_sum = 0.0

    for i in range(n):
        a = -x
        x = x + v * dt + 0.5 * a * dt * dt
        a_new = -x
        v = v + 0.5 * (a + a_new) * dt

        e = 0.5 * (v * v + x * x)
        energy_drift_sum += fabs(e - e0)

    return (x, v, energy_drift_sum)
