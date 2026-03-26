# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Gnome sort algorithm with swap counting (Cython-optimized).

Keywords: sorting, gnome sort, swap count, simple sort, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(3000,))
def gnome_sort(int n):
    """Sort a deterministic array using gnome sort, counting swaps."""
    cdef int *arr = <int *>malloc(n * sizeof(int))
    if not arr:
        raise MemoryError()

    cdef int i, pos, temp
    cdef long long swaps = 0
    cdef long long checksum = 0

    # Generate deterministic array
    for i in range(n):
        arr[i] = <int>((<unsigned int>i * <unsigned int>2654435761) % <unsigned int>n)

    pos = 0
    while pos < n:
        if pos == 0 or arr[pos] >= arr[pos - 1]:
            pos += 1
        else:
            temp = arr[pos]
            arr[pos] = arr[pos - 1]
            arr[pos - 1] = temp
            swaps += 1
            pos -= 1

    # Compute checksum: weighted sum of sorted positions
    for i in range(n):
        checksum += <long long>arr[i] * <long long>(i + 1)

    free(arr)
    return (int(checksum), int(swaps))
