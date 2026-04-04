# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Find min/max using struct return value.

Demonstrates struct return: cdef helper returns MinMax
struct with both values in one call.

Keywords: numerical, min, max, struct return, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


cdef struct MinMax:
    double min_val
    double max_val


cdef MinMax _find_minmax(
    double *arr, int n,
) noexcept:
    """Find min and max, return as struct."""
    cdef MinMax result
    cdef int i
    result.min_val = arr[0]
    result.max_val = arr[0]
    for i in range(1, n):
        if arr[i] < result.min_val:
            result.min_val = arr[i]
        if arr[i] > result.max_val:
            result.max_val = arr[i]
    return result


@cython_benchmark(syntax="cy", args=(100000,))
def struct_return_minmax(int n):
    """Find min/max of hash array, return sum."""
    cdef int i
    cdef unsigned int h

    cdef double *arr = <double *>malloc(
        n * sizeof(double)
    )
    if not arr:
        raise MemoryError()

    for i in range(n):
        h = (
            (<unsigned int>i
             * <unsigned int>2654435761)
            ^ (<unsigned int>i
               * <unsigned int>2246822519)
        )
        arr[i] = (
            <double>(h & 0xFFFF) / 65535.0
            * 200.0 - 100.0
        )

    cdef MinMax mm = _find_minmax(arr, n)

    free(arr)
    return mm.min_val + mm.max_val
