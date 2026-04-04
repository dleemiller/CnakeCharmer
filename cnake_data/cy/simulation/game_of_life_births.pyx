# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Conway's Game of Life with birth tracking (Cython-optimized).

Keywords: simulation, game of life, cellular automaton, grid, births, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100,))
def game_of_life_births(int n):
    """Run Game of Life for n generations on 80x80 grid, tracking births."""
    cdef int rows = 80
    cdef int cols = 80
    cdef int size = rows * cols
    cdef int i, j, di, dj, ni, nj, idx, neighbors, gen
    cdef int alive_count = 0
    cdef int total_births = 0
    cdef int center

    cdef char *current = <char *>malloc(size * sizeof(char))
    cdef char *nxt = <char *>malloc(size * sizeof(char))
    cdef char *temp_ptr

    if not current or not nxt:
        free(current); free(nxt)
        raise MemoryError()

    # Initialize grid with LCG pseudo-random (seed=42)
    cdef unsigned int lcg = 42
    for i in range(size):
        lcg = (lcg * 1103515245 + 12345) & 0x7FFFFFFF
        if lcg % 100 < 40:
            current[i] = 1
        else:
            current[i] = 0

    for gen in range(n):
        for i in range(rows):
            for j in range(cols):
                neighbors = 0
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        if di == 0 and dj == 0:
                            continue
                        ni = i + di
                        nj = j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            neighbors += current[ni * cols + nj]

                idx = i * cols + j
                if current[idx] == 1:
                    nxt[idx] = 1 if (neighbors == 2 or neighbors == 3) else 0
                else:
                    if neighbors == 3:
                        nxt[idx] = 1
                        total_births += 1
                    else:
                        nxt[idx] = 0

        # Swap pointers
        temp_ptr = current
        current = nxt
        nxt = temp_ptr

    alive_count = 0
    for i in range(size):
        alive_count += current[i]

    center = (rows // 2) * cols + (cols // 2)
    cdef int alive_at_center = current[center]

    free(current)
    free(nxt)
    return (alive_count, alive_at_center, total_births)
