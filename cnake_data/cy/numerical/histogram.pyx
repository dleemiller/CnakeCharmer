# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Count occurrences of values in a deterministically generated sequence (Cython-optimized).

Keywords: histogram, frequency, counting, numerical, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def histogram(int n):
    """Count value occurrences using C-typed loop and C array for bins."""
    cdef int i
    cdef int *counts = <int *>malloc(100 * sizeof(int))

    if counts == NULL:
        raise MemoryError("Failed to allocate array")

    memset(counts, 0, 100 * sizeof(int))

    for i in range(n):
        counts[(i * 31 + 17) % 100] += 1

    cdef list result = [counts[i] for i in range(100)]
    free(counts)
    return result
