# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Pancake sort using prefix reversals (Cython-optimized).

Keywords: sorting, pancake sort, prefix reversal, flips, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(3000,))
def pancake_sort(int n):
    """Sort a deterministic array using pancake sort (prefix reversals only)."""
    cdef int *arr = <int *>malloc(n * sizeof(int))
    if not arr:
        raise MemoryError()

    cdef int i, size, max_idx, lo, hi, temp
    cdef int flips = 0
    cdef long long checksum = 0

    # Generate deterministic array
    for i in range(n):
        arr[i] = <int>((<unsigned int>i * <unsigned int>2654435761) % <unsigned int>n)

    for size in range(n, 1, -1):
        # Find index of maximum element in arr[0..size-1]
        max_idx = 0
        for i in range(1, size):
            if arr[i] > arr[max_idx]:
                max_idx = i

        if max_idx == size - 1:
            continue

        # Flip to bring max to front
        if max_idx > 0:
            lo = 0
            hi = max_idx
            while lo < hi:
                temp = arr[lo]
                arr[lo] = arr[hi]
                arr[hi] = temp
                lo += 1
                hi -= 1
            flips += 1

        # Flip to put max at end of current range
        lo = 0
        hi = size - 1
        while lo < hi:
            temp = arr[lo]
            arr[lo] = arr[hi]
            arr[hi] = temp
            lo += 1
            hi -= 1
        flips += 1

    # Compute checksum: weighted sum of sorted positions
    for i in range(n):
        checksum += <long long>arr[i] * <long long>(i + 1)

    free(arr)
    return (int(checksum), flips)
