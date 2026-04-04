# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Forest fire simulation on a 2D grid (Cython-optimized).

Keywords: simulation, forest fire, cellular automaton, 2D grid, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200,))
def forest_fire(int n):
    """Simulate forest fire on n x n grid for 50 steps with C arrays."""
    cdef int steps = 50
    cdef int i, j, step, idx
    cdef int on_fire, tree_count
    cdef int nn = n * n

    cdef char *grid = <char *>malloc(nn * sizeof(char))
    cdef char *new_grid = <char *>malloc(nn * sizeof(char))
    cdef char *tmp

    if not grid or not new_grid:
        free(grid); free(new_grid)
        raise MemoryError()

    # Initialize
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            if (i * 17 + j * 11) % 3 != 0:
                grid[idx] = 1
            else:
                grid[idx] = 0
    grid[0] = 2  # Fire at (0,0)

    for step in range(steps):
        for i in range(n):
            for j in range(n):
                idx = i * n + j
                if grid[idx] == 2:
                    new_grid[idx] = 0
                elif grid[idx] == 1:
                    on_fire = 0
                    if i > 0 and grid[(i - 1) * n + j] == 2:
                        on_fire = 1
                    if i < n - 1 and grid[(i + 1) * n + j] == 2:
                        on_fire = 1
                    if j > 0 and grid[i * n + j - 1] == 2:
                        on_fire = 1
                    if j < n - 1 and grid[i * n + j + 1] == 2:
                        on_fire = 1
                    if on_fire:
                        new_grid[idx] = 2
                    else:
                        new_grid[idx] = 1
                else:
                    if (i * 7 + j * 13 + step * 31) % 100 < 1:
                        new_grid[idx] = 1
                    else:
                        new_grid[idx] = 0
        tmp = grid
        grid = new_grid
        new_grid = tmp

    tree_count = 0
    for i in range(nn):
        if grid[i] == 1:
            tree_count += 1

    free(grid)
    free(new_grid)
    return tree_count
