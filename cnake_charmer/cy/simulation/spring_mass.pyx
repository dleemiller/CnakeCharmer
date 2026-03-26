# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""1D spring-mass chain simulation (Cython-optimized).

Keywords: simulation, spring, mass, physics, Hooke's law, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(3000,))
def spring_mass(int n):
    """Simulate n masses connected by springs in 1D for 1000 timesteps.

    Args:
        n: Number of masses.

    Returns:
        Sum of final positions.
    """
    cdef int timesteps = 1000
    cdef double k = 1.0
    cdef double dt = 0.01
    cdef int i, t
    cdef double force, total

    cdef double *x = <double *>malloc(n * sizeof(double))
    cdef double *v = <double *>malloc(n * sizeof(double))
    if not x or not v:
        free(x)
        free(v)
        raise MemoryError()

    for i in range(n):
        x[i] = i * 1.0
        v[i] = 0.0

    for t in range(timesteps):
        # Compute forces and update velocities
        for i in range(n):
            force = 0.0
            if i > 0:
                force += k * (x[i - 1] - x[i])
            if i < n - 1:
                force += k * (x[i + 1] - x[i])
            v[i] += force * dt
        # Update positions
        for i in range(n):
            x[i] += v[i] * dt

    total = 0.0
    for i in range(n):
        total += x[i]

    free(x)
    free(v)
    return total
