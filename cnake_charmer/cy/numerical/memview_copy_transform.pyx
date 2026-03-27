# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Copy a typed memoryview, apply an in-place transform to the copy, and return the sum.

Keywords: numerical, copy, transform, array, typed memoryview, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def memview_copy_transform(int n):
    """Copy memoryview with .copy(), square each element, return sum."""
    cdef int i
    cdef double total

    cdef double *ptr = <double *>malloc(n * sizeof(double))
    if not ptr:
        raise MemoryError()

    cdef double[::1] data = <double[:n]>ptr

    for i in range(n):
        data[i] = ((i * 29 + 5) % 200) / 10.0

    # Use memoryview .copy() to create an independent copy
    cdef double[::1] copy = data.copy()

    # Transform: square each element in the copy
    for i in range(n):
        copy[i] = copy[i] * copy[i]

    total = 0.0
    for i in range(n):
        total += copy[i]

    free(ptr)
    return total
