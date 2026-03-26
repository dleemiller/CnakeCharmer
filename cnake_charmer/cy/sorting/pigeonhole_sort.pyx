# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Pigeonhole sort algorithm (Cython-optimized).

Keywords: sorting, pigeonhole sort, counting, distribution, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000000,))
def pigeonhole_sort(int n):
    """Sort a deterministic array using pigeonhole sort."""
    cdef int *arr = <int *>malloc(n * sizeof(int))
    if not arr:
        raise MemoryError()

    cdef int holes[1000]
    memset(holes, 0, 1000 * sizeof(int))

    cdef int i, j, idx

    # Generate deterministic array
    for i in range(n):
        arr[i] = (i * 31 + 17) % 1000

    # Count occurrences
    for i in range(n):
        holes[arr[i]] += 1

    # Build sorted output
    cdef int *output = <int *>malloc(n * sizeof(int))
    if not output:
        free(arr)
        raise MemoryError()

    idx = 0
    for i in range(1000):
        for j in range(holes[i]):
            output[idx] = i
            idx += 1

    cdef list result = [output[i] for i in range(n)]
    free(arr)
    free(output)
    return result
