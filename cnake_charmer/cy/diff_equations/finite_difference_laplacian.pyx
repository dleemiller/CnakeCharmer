# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Solve 2D Laplace equation on n x n grid using Jacobi iteration (Cython-optimized).

Boundary conditions: top=100, others=0. Returns sum of interior values.

Keywords: PDE, Laplace equation, Jacobi iteration, finite difference, numerical, cython
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100,))
def finite_difference_laplacian(int n):
    """Solve 2D Laplace equation using Jacobi iteration."""
    cdef int i, j, k, num_iters, nn
    cdef double total
    cdef double *grid
    cdef double *new_grid
    cdef double *temp

    num_iters = 500
    nn = n * n

    grid = <double *>malloc(nn * sizeof(double))
    new_grid = <double *>malloc(nn * sizeof(double))
    if not grid or not new_grid:
        if grid:
            free(grid)
        if new_grid:
            free(new_grid)
        raise MemoryError()

    # Initialize to zero
    for i in range(nn):
        grid[i] = 0.0
        new_grid[i] = 0.0

    # Set top boundary to 100
    for j in range(n):
        grid[j] = 100.0
        new_grid[j] = 100.0

    # Jacobi iteration
    for k in range(num_iters):
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                new_grid[i * n + j] = 0.25 * (
                    grid[(i - 1) * n + j] +
                    grid[(i + 1) * n + j] +
                    grid[i * n + (j - 1)] +
                    grid[i * n + (j + 1)]
                )
        # Swap pointers
        temp = grid
        grid = new_grid
        new_grid = temp

    # Sum interior values
    total = 0.0
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            total += grid[i * n + j]

    free(grid)
    free(new_grid)
    return total
