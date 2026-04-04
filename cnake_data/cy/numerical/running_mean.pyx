# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Compute running mean of a deterministically generated sequence (Cython-optimized).

Keywords: running mean, cumulative average, numerical, statistics, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def running_mean(int n):
    """Compute the running mean using C-typed cumulative sum."""
    cdef double cumsum = 0.0
    cdef double value
    cdef int i
    cdef double *arr = <double *>malloc(n * sizeof(double))

    if arr == NULL:
        raise MemoryError("Failed to allocate array")

    for i in range(n):
        value = ((i * 7 + 3) % 1000) / 10.0
        cumsum += value
        arr[i] = cumsum / (i + 1)

    cdef list result = [arr[i] for i in range(n)]

    free(arr)
    return result
