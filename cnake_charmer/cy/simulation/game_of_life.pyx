# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Conway's Game of Life simulation (Cython-optimized).

Keywords: game of life, cellular automaton, simulation, grid, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(120,))
def game_of_life(int n):
    """Run Conway's Game of Life on an n x n grid for 50 generations."""
    cdef int generations = 50
    cdef int size = n * n
    cdef int *current = <int *>malloc(size * sizeof(int))
    cdef int *nxt = <int *>malloc(size * sizeof(int))
    cdef int *tmp
    if not current or not nxt:
        free(current); free(nxt)
        raise MemoryError()

    cdef int i, j, di, dj, ni, nj, neighbors, idx, gen, total
    cdef long cell_checksum = 0

    # Initialize
    for i in range(n):
        for j in range(n):
            if (i * 7 + j * 13 + 3) % 5 == 0:
                current[i * n + j] = 1
            else:
                current[i * n + j] = 0

    for gen in range(generations):
        memset(nxt, 0, size * sizeof(int))
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
                            neighbors += current[ni * n + nj]

                idx = i * n + j
                if current[idx] == 1:
                    if neighbors == 2 or neighbors == 3:
                        nxt[idx] = 1
                else:
                    if neighbors == 3:
                        nxt[idx] = 1

        # Swap buffers
        tmp = current
        current = nxt
        nxt = tmp

    total = 0
    cell_checksum = 0
    for i in range(size):
        total += current[i]
        cell_checksum += current[i] * ((i * 31 + 7) % 10000)

    free(current)
    free(nxt)
    return (total, cell_checksum)
