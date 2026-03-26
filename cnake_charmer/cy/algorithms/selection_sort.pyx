# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Selection sort (Cython-optimized with C array).

Keywords: sorting, selection sort, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000,))
def selection_sort(int n):
    """Sort a reversed list using C-level selection sort with malloc'd array."""
    cdef int *arr = <int *>malloc(n * sizeof(int))
    if not arr:
        raise MemoryError()

    cdef int i, j, min_idx, tmp

    # Fill reversed
    for i in range(n):
        arr[i] = n - i

    # Selection sort on C array
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        tmp = arr[i]
        arr[i] = arr[min_idx]
        arr[min_idx] = tmp

    # Convert back to Python list
    cdef list result = [arr[i] for i in range(n)]
    free(arr)
    return result
