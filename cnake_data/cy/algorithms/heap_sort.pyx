# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Heap sort on a deterministic integer array (Cython-optimized).

Keywords: algorithms, heap sort, sorting, in-place, sift-down, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200000,))
def heap_sort(int n):
    """Sort array using heap sort with C array and sift-down."""
    cdef int *arr = <int *>malloc(n * sizeof(int))
    if not arr:
        raise MemoryError()

    cdef int i, pos, child, end, tmp

    # Generate input
    for i in range(n):
        arr[i] = (i * 31 + 17) % n

    # Build max-heap (sift-down from last parent to root)
    for i in range(n / 2 - 1, -1, -1):
        pos = i
        while True:
            child = 2 * pos + 1
            if child >= n:
                break
            if child + 1 < n and arr[child + 1] > arr[child]:
                child += 1
            if arr[child] > arr[pos]:
                tmp = arr[pos]
                arr[pos] = arr[child]
                arr[child] = tmp
                pos = child
            else:
                break

    # Extract elements from heap
    end = n - 1
    while end > 0:
        tmp = arr[0]
        arr[0] = arr[end]
        arr[end] = tmp
        # Sift down in reduced heap
        pos = 0
        while True:
            child = 2 * pos + 1
            if child >= end:
                break
            if child + 1 < end and arr[child + 1] > arr[child]:
                child += 1
            if arr[child] > arr[pos]:
                tmp = arr[pos]
                arr[pos] = arr[child]
                arr[child] = tmp
                pos = child
            else:
                break
        end -= 1

    # Convert to Python list
    cdef list result = [arr[i] for i in range(n)]
    free(arr)
    return result
