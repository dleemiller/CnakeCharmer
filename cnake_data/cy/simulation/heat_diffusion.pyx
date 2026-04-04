# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""1D heat equation simulation using finite differences (Cython-optimized).

Keywords: heat equation, diffusion, finite difference, simulation, PDE, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin, M_PI
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000,))
def heat_diffusion(int n):
    """Simulate 1D heat equation using finite differences."""
    cdef int timesteps = 1000
    cdef double alpha = 0.1
    cdef int i, t
    cdef double total

    cdef double *u = <double *>malloc(n * sizeof(double))
    cdef double *u_new = <double *>malloc(n * sizeof(double))
    cdef double *tmp
    if not u or not u_new:
        free(u); free(u_new)
        raise MemoryError()

    # Initialize
    for i in range(n):
        u[i] = sin(i * M_PI / n) * 100.0
    u[0] = 0.0
    u[n - 1] = 0.0

    for t in range(timesteps):
        u_new[0] = 0.0
        u_new[n - 1] = 0.0
        for i in range(1, n - 1):
            u_new[i] = u[i] + alpha * (u[i - 1] - 2.0 * u[i] + u[i + 1])
        tmp = u
        u = u_new
        u_new = tmp

    total = 0.0
    for i in range(n):
        total += u[i]

    free(u)
    free(u_new)
    return total
