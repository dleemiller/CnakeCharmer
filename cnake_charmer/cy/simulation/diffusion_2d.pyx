# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""2D diffusion simulation on a grid using Laplacian stencil (Cython-optimized).

Keywords: simulation, diffusion, 2D, Laplacian, PDE, finite difference, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(150,))
def diffusion_2d(int n):
    """Simulate 2D diffusion on an n x n grid for 100 timesteps.

    Args:
        n: Grid dimension (n x n).

    Returns:
        Sum of all grid values after simulation.
    """
    cdef int timesteps = 100
    cdef double alpha = 0.1
    cdef int i, j, t, idx, size
    cdef double total, laplacian

    size = n * n

    cdef double *u = <double *>malloc(size * sizeof(double))
    cdef double *u_new = <double *>malloc(size * sizeof(double))
    cdef double *tmp
    if not u or not u_new:
        free(u)
        free(u_new)
        raise MemoryError()

    memset(u, 0, size * sizeof(double))
    memset(u_new, 0, size * sizeof(double))
    u[n * (n // 2) + n // 2] = 1.0

    for t in range(timesteps):
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                idx = i * n + j
                laplacian = (u[(i - 1) * n + j] + u[(i + 1) * n + j]
                             + u[i * n + j - 1] + u[i * n + j + 1]
                             - 4.0 * u[idx])
                u_new[idx] = u[idx] + alpha * laplacian
        # Swap pointers
        tmp = u
        u = u_new
        u_new = tmp
        # Zero boundary in u_new for next iteration
        for i in range(n):
            u_new[i] = 0.0
            u_new[(n - 1) * n + i] = 0.0
            u_new[i * n] = 0.0
            u_new[i * n + n - 1] = 0.0

    total = 0.0
    for i in range(size):
        total += u[i]

    free(u)
    free(u_new)
    return total
