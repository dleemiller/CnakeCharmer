# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Comb sort algorithm (Cython-optimized).

Keywords: sorting, comb sort, gap, shrink factor, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def comb_sort(int n):
    """Sort a deterministic array using comb sort."""
    cdef int *arr = <int *>malloc(n * sizeof(int))
    if not arr:
        raise MemoryError()

    cdef int i, gap, temp
    cdef int sorted_flag
    cdef double shrink = 1.3

    # Generate deterministic array
    for i in range(n):
        arr[i] = (i * 31 + 17) % n

    gap = n
    sorted_flag = 0

    while sorted_flag == 0:
        gap = <int>(gap / shrink)
        if gap <= 1:
            gap = 1
            sorted_flag = 1
        else:
            sorted_flag = 0

        for i in range(n - gap):
            if arr[i] > arr[i + gap]:
                temp = arr[i]
                arr[i] = arr[i + gap]
                arr[i + gap] = temp
                sorted_flag = 0

    cdef list result = [arr[i] for i in range(n)]
    free(arr)
    return result
