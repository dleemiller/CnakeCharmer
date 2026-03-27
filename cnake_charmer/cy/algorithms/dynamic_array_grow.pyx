# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Dynamically grow array using realloc.

Keywords: algorithms, dynamic array, realloc, grow, cython, benchmark
"""

from libc.stdlib cimport malloc, realloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def dynamic_array_grow(int n):
    """Push n values using realloc to double capacity."""
    cdef int capacity = 16
    cdef int size = 0
    cdef long long *arr = <long long *>malloc(
        capacity * sizeof(long long)
    )
    if not arr:
        raise MemoryError()

    cdef int i
    cdef long long val, total
    cdef long long *tmp

    for i in range(n):
        val = (
            (
                <long long>i * <long long>2654435761 + 17
            ) ^ (
                <long long>i * <long long>1103515245
            )
        ) & 0x7FFFFFFF
        if size >= capacity:
            capacity *= 2
            tmp = <long long *>realloc(
                arr, capacity * sizeof(long long)
            )
            if not tmp:
                free(arr)
                raise MemoryError()
            arr = tmp
        arr[size] = val
        size += 1

    total = 0
    for i in range(size):
        total += arr[i]

    free(arr)
    return total
