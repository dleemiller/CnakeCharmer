# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Abelian sandpile model on a 2D grid (Cython-optimized).

Keywords: simulation, sandpile, abelian, toppling, cellular automaton, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200,))
def sandpile(int n):
    """Simulate abelian sandpile on n x n grid with C arrays."""
    cdef int i, j, idx, center, spill
    cdef int nn = n * n
    cdef int changed, count

    cdef int *grid = <int *>malloc(nn * sizeof(int))
    if not grid:
        raise MemoryError()

    memset(grid, 0, nn * sizeof(int))
    center = n // 2
    grid[center * n + center] = 4 * n

    changed = 1
    while changed:
        changed = 0
        for i in range(n):
            for j in range(n):
                idx = i * n + j
                if grid[idx] >= 4:
                    changed = 1
                    spill = grid[idx] // 4
                    grid[idx] = grid[idx] % 4
                    if i > 0:
                        grid[(i - 1) * n + j] += spill
                    if i < n - 1:
                        grid[(i + 1) * n + j] += spill
                    if j > 0:
                        grid[i * n + j - 1] += spill
                    if j < n - 1:
                        grid[i * n + j + 1] += spill

    count = 0
    for i in range(nn):
        if grid[i] >= 1:
            count += 1

    free(grid)
    return count
