# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Greedy maximum independent set on a deterministic graph (Cython-optimized).

Keywords: graph, independent set, greedy, optimization, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def max_independent_set(int n):
    """Compute greedy maximum independent set size with C arrays."""
    cdef int i, j1, j2
    cdef int count = 0

    cdef char *excluded = <char *>malloc(n * sizeof(char))
    if not excluded:
        raise MemoryError()

    memset(excluded, 0, n * sizeof(char))

    for i in range(n):
        if excluded[i] == 0:
            count += 1
            j1 = (i * 3 + 1) % n
            j2 = (i * 7 + 2) % n
            excluded[j1] = 1
            excluded[j2] = 1

    free(excluded)
    return count
