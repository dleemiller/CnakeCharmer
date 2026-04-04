# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute exponentially weighted moving average (Cython-optimized).

Keywords: ewma, exponential, moving average, smoothing, numerical, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def ewma(int n):
    """Compute EWMA with alpha=0.1 using C array for computation."""
    cdef double alpha = 0.1
    cdef double one_minus_alpha = 0.9
    cdef double avg, value
    cdef int i
    cdef double *arr = <double *>malloc(n * sizeof(double))

    if arr == NULL:
        raise MemoryError()

    avg = ((0 * 7 + 3) % 1000) / 10.0
    arr[0] = avg

    for i in range(1, n):
        value = ((i * 7 + 3) % 1000) / 10.0
        avg = alpha * value + one_minus_alpha * avg
        arr[i] = avg

    cdef list result = [arr[i] for i in range(n)]
    free(arr)
    return result
