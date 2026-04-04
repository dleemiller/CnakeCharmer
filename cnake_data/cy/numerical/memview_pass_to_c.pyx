# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Compute weighted sum by passing memoryview data to a cdef C function via pointer.

Keywords: numerical, weighted sum, interop, array, typed memoryview, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


cdef double _weighted_sum(double *data, int n):
    """C-level function that receives a raw pointer from a memoryview."""
    cdef int i
    cdef double result = 0.0
    for i in range(n):
        result += data[i] * <double>(i + 1)
    return result


@cython_benchmark(syntax="cy", args=(100000,))
def memview_pass_to_c(int n):
    """Create memoryview, pass to cdef function via &view[0], return weighted sum."""
    cdef int i

    cdef double *ptr = <double *>malloc(n * sizeof(double))
    if not ptr:
        raise MemoryError()

    cdef double[::1] data = <double[:n]>ptr

    for i in range(n):
        data[i] = ((i * 67 + 23) % 300) / 15.0

    # Pass memoryview to C function via &data[0]
    cdef double result = _weighted_sum(&data[0], n)

    free(ptr)
    return result
