# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Langton's ant simulation (Cython-optimized).

Keywords: langton, ant, cellular automaton, simulation, grid, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500,))
def langtons_ant(int n):
    """Simulate Langton's ant using C unsigned char array."""
    cdef int size = n * n
    cdef unsigned char *grid = <unsigned char *>malloc(size * sizeof(unsigned char))
    if not grid:
        raise MemoryError()

    memset(grid, 0, size * sizeof(unsigned char))

    cdef int x = n // 2
    cdef int y = n // 2
    cdef int direction = 0  # 0=up, 1=right, 2=down, 3=left
    cdef int steps = size
    cdef int idx, i, count

    # Direction deltas: dx[d], dy[d]
    cdef int dx[4]
    cdef int dy[4]
    dx[0] = 0; dx[1] = 1; dx[2] = 0; dx[3] = -1
    dy[0] = -1; dy[1] = 0; dy[2] = 1; dy[3] = 0

    for i in range(steps):
        idx = y * n + x
        if grid[idx] == 0:
            direction = (direction + 1) % 4
            grid[idx] = 1
        else:
            direction = (direction + 3) % 4
            grid[idx] = 0
        x = (x + dx[direction] + n) % n
        y = (y + dy[direction] + n) % n

    count = 0
    for i in range(size):
        count += grid[i]

    free(grid)
    return count
