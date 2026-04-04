# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Bubble sort (Cython-optimized with C array).

Keywords: sorting, bubble sort, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(3000,))
def bubble_sort(int n):
    """Sort a reversed list using C-level bubble sort with malloc'd array."""
    cdef int *arr = <int *>malloc(n * sizeof(int))
    if not arr:
        raise MemoryError()

    cdef int i, j, tmp

    # Fill reversed
    for i in range(n):
        arr[i] = n - i

    # Bubble sort on C array
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                tmp = arr[j]
                arr[j] = arr[j + 1]
                arr[j + 1] = tmp

    # Convert back to Python list
    cdef list result = [arr[i] for i in range(n)]
    free(arr)
    return result
