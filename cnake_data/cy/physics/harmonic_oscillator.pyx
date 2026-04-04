# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Simulate coupled harmonic oscillators using Verlet integration (Cython-optimized).

Keywords: physics, harmonic, oscillator, verlet, simulation, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(3000,))
def harmonic_oscillator(int n):
    """Simulate coupled oscillators using C arrays and Verlet integration."""
    cdef double *x = <double *>malloc(n * sizeof(double))
    cdef double *v = <double *>malloc(n * sizeof(double))
    cdef double *a = <double *>malloc(n * sizeof(double))
    cdef double *a_new = <double *>malloc(n * sizeof(double))

    if not x or not v or not a or not a_new:
        if x: free(x)
        if v: free(v)
        if a: free(a)
        if a_new: free(a_new)
        raise MemoryError()

    cdef double k = 1.0
    cdef double dt = 0.001
    cdef int steps = 1000
    cdef int i, step
    cdef double force, energy, dx

    # Initialize
    for i in range(n):
        x[i] = sin(i * 0.1)
        v[i] = 0.0

    # Initial accelerations
    for i in range(n):
        force = -k * x[i]
        if i > 0:
            force += k * (x[i - 1] - x[i])
        if i < n - 1:
            force += k * (x[i + 1] - x[i])
        a[i] = force

    # Verlet integration
    for step in range(steps):
        for i in range(n):
            x[i] = x[i] + v[i] * dt + 0.5 * a[i] * dt * dt

        for i in range(n):
            force = -k * x[i]
            if i > 0:
                force += k * (x[i - 1] - x[i])
            if i < n - 1:
                force += k * (x[i + 1] - x[i])
            a_new[i] = force

        for i in range(n):
            v[i] = v[i] + 0.5 * (a[i] + a_new[i]) * dt
            a[i] = a_new[i]

    # Compute total energy
    energy = 0.0
    for i in range(n):
        energy += 0.5 * v[i] * v[i]
        energy += 0.5 * k * x[i] * x[i]
        if i < n - 1:
            dx = x[i + 1] - x[i]
            energy += 0.5 * k * dx * dx

    free(x)
    free(v)
    free(a)
    free(a_new)
    return energy
