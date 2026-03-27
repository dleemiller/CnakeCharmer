# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Simulate a spring-damper (mass-spring-dashpot) system (Cython-optimized).

Keywords: physics, spring, damper, dashpot, oscillation, simulation, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def spring_damper(int n):
    """Simulate a spring-damper system for n timesteps."""
    cdef double m = 1.0
    cdef double k = 50.0
    cdef double c = 0.5
    cdef double dt = 1e-5
    cdef double x = 2.0
    cdef double v = 0.0
    cdef double max_disp = 2.0
    cdef double a, ax
    cdef int i

    for i in range(n):
        a = (-k * x - c * v) / m
        v = v + a * dt
        x = x + v * dt
        ax = x if x >= 0.0 else -x
        if ax > max_disp:
            max_disp = ax

    return (x, v, max_disp)
