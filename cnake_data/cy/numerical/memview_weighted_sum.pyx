# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Weighted prefix sum with exponential decay (Cython with typed memoryviews).

Keywords: weighted sum, prefix sum, typed memoryview, exponential decay, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def memview_weighted_sum(int n):
    """Compute weighted prefix sums using typed memoryviews."""
    cdef double *val_ptr = <double *>malloc(n * sizeof(double))
    cdef double *res_ptr = <double *>malloc(n * sizeof(double))
    if not val_ptr or not res_ptr:
        if val_ptr: free(val_ptr)
        if res_ptr: free(res_ptr)
        raise MemoryError()

    # Create typed memoryviews from raw pointers
    cdef double[:] values = <double[:n]>val_ptr
    cdef double[:] result = <double[:n]>res_ptr

    cdef double decay = 0.999
    cdef int i
    cdef double total = 0.0
    cdef long long htmp

    for i in range(n):
        htmp = (<long long>i * <long long>2654435761) % 1000
        values[i] = htmp / 100.0

    # Compute using recurrence with memoryview access
    result[0] = values[0]
    for i in range(1, n):
        result[i] = values[i] + decay * result[i - 1]

    for i in range(n):
        total += result[i]

    free(val_ptr)
    free(res_ptr)
    return total
