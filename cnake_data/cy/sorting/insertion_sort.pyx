# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Insertion sort (Cython-optimized with C array).

Keywords: sorting, insertion sort, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000,))
def insertion_sort(int n):
    """Sort a reversed list using C-level insertion sort with malloc'd array."""
    cdef int *arr = <int *>malloc(n * sizeof(int))
    if not arr:
        raise MemoryError()

    cdef int i, j, key

    # Fill reversed
    for i in range(n):
        arr[i] = n - i

    # Insertion sort on C array
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

    # Convert back to Python list
    cdef list result = [arr[i] for i in range(n)]
    free(arr)
    return result
