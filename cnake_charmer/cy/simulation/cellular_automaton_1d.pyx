# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Rule 110 one-dimensional cellular automaton (Cython-optimized).

Keywords: simulation, cellular automaton, rule 110, 1D, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(10000,))
def cellular_automaton_1d(int n):
    """Run Rule 110 on n cells for 500 generations.

    Args:
        n: Number of cells.

    Returns:
        Count of live (1) cells after 500 generations.
    """
    cdef int generations = 500
    cdef int i, g, left, center, right, pattern, count
    cdef unsigned char rule[8]
    rule[0] = 0; rule[1] = 1; rule[2] = 1; rule[3] = 1
    rule[4] = 0; rule[5] = 1; rule[6] = 1; rule[7] = 0

    cdef unsigned char *cells = <unsigned char *>malloc(n * sizeof(unsigned char))
    cdef unsigned char *new_cells = <unsigned char *>malloc(n * sizeof(unsigned char))
    cdef unsigned char *tmp
    if not cells or not new_cells:
        free(cells)
        free(new_cells)
        raise MemoryError()

    memset(cells, 0, n * sizeof(unsigned char))
    cells[n // 2] = 1

    for g in range(generations):
        for i in range(n):
            left = cells[(i - 1 + n) % n]
            center = cells[i]
            right = cells[(i + 1) % n]
            pattern = (left << 2) | (center << 1) | right
            new_cells[i] = rule[pattern]
        tmp = cells
        cells = new_cells
        new_cells = tmp

    count = 0
    for i in range(n):
        count += cells[i]

    free(cells)
    free(new_cells)
    return count
