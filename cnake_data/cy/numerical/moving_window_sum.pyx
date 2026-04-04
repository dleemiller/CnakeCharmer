# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Compute sliding window sum over a deterministically generated sequence (Cython-optimized).

Keywords: numerical, sliding window, moving sum, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def moving_window_sum(int n):
    """Compute sliding window sum of size 100 using C arrays."""
    cdef int window = 100
    cdef int out_len
    cdef int i
    cdef int current_sum
    cdef int *values
    cdef int *out

    if n < window:
        return []

    out_len = n - window + 1

    values = <int *>malloc(n * sizeof(int))
    out = <int *>malloc(out_len * sizeof(int))

    if values == NULL or out == NULL:
        if values != NULL:
            free(values)
        if out != NULL:
            free(out)
        raise MemoryError("Failed to allocate arrays")

    for i in range(n):
        values[i] = (i * 13 + 7) % 1000

    current_sum = 0
    for i in range(window):
        current_sum += values[i]

    out[0] = current_sum
    for i in range(window, n):
        current_sum += values[i] - values[i - window]
        out[i - window + 1] = current_sum

    cdef list result = [out[i] for i in range(out_len)]

    free(values)
    free(out)
    return result
