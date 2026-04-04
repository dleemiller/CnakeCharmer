# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""1D wave equation simulation using finite differences (Cython-optimized).

Keywords: wave equation, simulation, PDE, finite difference, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin, M_PI
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000,))
def wave_equation(int n):
    """Simulate 1D wave equation on n cells for 500 timesteps."""
    cdef int timesteps = 500
    cdef double c = 1.0
    cdef double dt = 0.01
    cdef double dx = 1.0 / n
    cdef double r = (c * dt / dx) * (c * dt / dx)
    cdef int i, t
    cdef double total

    cdef double *u = <double *>malloc(n * sizeof(double))
    cdef double *u_prev = <double *>malloc(n * sizeof(double))
    cdef double *u_next = <double *>malloc(n * sizeof(double))
    cdef double *tmp
    if not u or not u_prev or not u_next:
        free(u); free(u_prev); free(u_next)
        raise MemoryError()

    for i in range(n):
        u[i] = sin(i * M_PI / n) * 100.0
        u_prev[i] = sin(i * M_PI / n) * 100.0
        u_next[i] = 0.0
    u[0] = 0.0
    u[n - 1] = 0.0
    u_prev[0] = 0.0
    u_prev[n - 1] = 0.0

    for t in range(timesteps):
        u_next[0] = 0.0
        u_next[n - 1] = 0.0
        for i in range(1, n - 1):
            u_next[i] = (
                2.0 * u[i]
                - u_prev[i]
                + r * (u[i - 1] - 2.0 * u[i] + u[i + 1])
            )
        tmp = u_prev
        u_prev = u
        u = u_next
        u_next = tmp

    total = 0.0
    for i in range(n):
        total += u[i]

    free(u)
    free(u_prev)
    free(u_next)
    return total
