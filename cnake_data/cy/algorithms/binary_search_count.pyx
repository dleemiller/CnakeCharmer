# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Binary search count (Cython-optimized with C arrays).

Keywords: algorithms, binary search, searching, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def binary_search_count(int n):
    """Count query values found in sorted array using binary search on C arrays."""
    cdef int *arr = <int *>malloc(n * sizeof(int))
    if not arr:
        raise MemoryError()

    cdef int i, j, target, lo, hi, mid, count, last_found_idx

    # Build sorted array: arr[i] = i * 3
    for i in range(n):
        arr[i] = i * 3

    count = 0
    last_found_idx = -1
    for j in range(n):
        target = j * 5
        lo = 0
        hi = n - 1
        while lo <= hi:
            mid = (lo + hi) / 2
            if arr[mid] == target:
                count += 1
                last_found_idx = mid
                break
            elif arr[mid] < target:
                lo = mid + 1
            else:
                hi = mid - 1

    free(arr)
    return (count, last_found_idx)
