# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Conway's Game of Life on an n*n grid for 50 steps using 2D typed memoryviews, return live cell count.

Keywords: simulation, cellular automaton, game of life, 2D, typed memoryview, cython, benchmark
"""

from cython.view cimport array as cvarray
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200,))
def memview_game_of_life(int n):
    """Run Game of Life for 50 steps using 2D memoryviews, return live cell count."""
    cdef int steps = 50
    cdef int i, j, di, dj, ni, nj, neighbors, alive, count, step

    arr_grid = cvarray(shape=(n, n), itemsize=sizeof(int), format="i")
    cdef int[:, :] grid = arr_grid

    arr_buf = cvarray(shape=(n, n), itemsize=sizeof(int), format="i")
    cdef int[:, :] buf = arr_buf

    # Initialize grid
    for i in range(n):
        for j in range(n):
            if ((i * 71 + j * 43 + 19) % 7) < 2:
                grid[i, j] = 1
            else:
                grid[i, j] = 0

    for step in range(steps):
        for i in range(n):
            for j in range(n):
                neighbors = 0
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        if di == 0 and dj == 0:
                            continue
                        ni = i + di
                        nj = j + dj
                        if 0 <= ni < n and 0 <= nj < n:
                            neighbors += grid[ni, nj]
                alive = grid[i, j]
                if alive:
                    buf[i, j] = 1 if (neighbors == 2 or neighbors == 3) else 0
                else:
                    buf[i, j] = 1 if neighbors == 3 else 0

        # Swap views
        grid, buf = buf, grid

    count = 0
    for i in range(n):
        for j in range(n):
            count += grid[i, j]

    return count
