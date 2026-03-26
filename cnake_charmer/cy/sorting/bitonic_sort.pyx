# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Bitonic sort network for power-of-2 arrays (Cython-optimized).

Keywords: sorting, bitonic sort, sorting network, parallel sort, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(32768,))
def bitonic_sort(int n):
    """Sort a deterministic array using bitonic sort network."""
    cdef int *arr = <int *>malloc(n * sizeof(int))
    if not arr:
        raise MemoryError()

    cdef int i, j, k, l, temp
    cdef long long comparisons = 0
    cdef long long checksum = 0

    # Generate deterministic array
    for i in range(n):
        arr[i] = <int>((<unsigned int>i * <unsigned int>2654435761) % <unsigned int>n)

    # Bitonic sort network
    k = 2
    while k <= n:
        j = k >> 1
        while j > 0:
            for i in range(n):
                l = i ^ j
                if l > i:
                    # Ascending if (i & k) == 0, descending otherwise
                    if (i & k) == 0:
                        if arr[i] > arr[l]:
                            temp = arr[i]
                            arr[i] = arr[l]
                            arr[l] = temp
                    else:
                        if arr[i] < arr[l]:
                            temp = arr[i]
                            arr[i] = arr[l]
                            arr[l] = temp
                    comparisons += 1
            j >>= 1
        k <<= 1

    # Compute checksum: weighted sum of sorted positions
    for i in range(n):
        checksum += <long long>arr[i] * <long long>(i + 1)

    free(arr)
    return (int(checksum), int(comparisons))
