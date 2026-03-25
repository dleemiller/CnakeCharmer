# cython: boundscheck=False, wraparound=False, language_level=3
"""
Bubble sort (Cython-optimized).

Keywords: sorting, bubble sort, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark
import cython


@cython_benchmark(syntax="cy", args=(3000,))
def bubble_sort(int n):
    """Sort a reversed list using typed bubble sort."""
    cdef list arr = list(range(n, 0, -1))
    cdef int i, j, tmp

    for i in range(n):
        for j in range(0, n - i - 1):
            if <int>arr[j] > <int>arr[j + 1]:
                tmp = <int>arr[j]
                arr[j] = arr[j + 1]
                arr[j + 1] = tmp

    return arr
