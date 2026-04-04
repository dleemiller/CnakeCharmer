# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Cocktail shaker sort algorithm (Cython).

Keywords: sorting, cocktail sort, bidirectional bubble sort, shaker sort, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(3000,))
def cocktail_sort(int n):
    """Sort a deterministic array using cocktail shaker sort."""
    cdef int *arr = <int *>malloc(n * sizeof(int))
    if not arr:
        raise MemoryError()

    cdef int i, start, end, temp
    cdef int swapped

    for i in range(n):
        arr[i] = (i * 47 + 13) % n

    start = 0
    end = n - 1
    swapped = 1

    while swapped:
        swapped = 0

        # Forward pass
        for i in range(start, end):
            if arr[i] > arr[i + 1]:
                temp = arr[i]
                arr[i] = arr[i + 1]
                arr[i + 1] = temp
                swapped = 1

        if not swapped:
            break

        end -= 1
        swapped = 0

        # Backward pass
        for i in range(end, start, -1):
            if arr[i] < arr[i - 1]:
                temp = arr[i]
                arr[i] = arr[i - 1]
                arr[i - 1] = temp
                swapped = 1

        start += 1

    cdef list result = [arr[i] for i in range(n)]
    free(arr)
    return result
