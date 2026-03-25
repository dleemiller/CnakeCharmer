# cython: boundscheck=False, wraparound=False, language_level=3
"""
Insertion sort (Cython-optimized).

Keywords: sorting, insertion sort, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark
import cython


@cython_benchmark(syntax="cy", args=(5000,))
def insertion_sort(int n):
    """Sort a reversed list using typed insertion sort."""
    cdef list arr = list(range(n, 0, -1))
    cdef int i, j, key
    cdef int length = len(arr)

    for i in range(1, length):
        key = <int>arr[i]
        j = i - 1
        while j >= 0 and <int>arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

    return arr
