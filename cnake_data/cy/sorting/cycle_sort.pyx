# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Cycle sort algorithm (Cython-optimized).

Keywords: sorting, cycle sort, minimal writes, in-place, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(3000,))
def cycle_sort(int n):
    """Count the number of writes performed by cycle sort."""
    cdef int *arr = <int *>malloc(n * sizeof(int))
    if not arr:
        raise MemoryError()

    cdef int i, cycle_start, pos, item, temp
    cdef int writes = 0

    # Generate deterministic array
    for i in range(n):
        arr[i] = (i * 31 + 17) % n

    for cycle_start in range(n - 1):
        item = arr[cycle_start]

        # Find the position where item should go
        pos = cycle_start
        for i in range(cycle_start + 1, n):
            if arr[i] < item:
                pos += 1

        # If item is already in correct position, skip
        if pos == cycle_start:
            continue

        # Skip duplicates
        while item == arr[pos]:
            pos += 1

        # Put the item in its correct position
        temp = arr[pos]
        arr[pos] = item
        item = temp
        writes += 1

        # Rotate the rest of the cycle
        while pos != cycle_start:
            pos = cycle_start
            for i in range(cycle_start + 1, n):
                if arr[i] < item:
                    pos += 1

            while item == arr[pos]:
                pos += 1

            temp = arr[pos]
            arr[pos] = item
            item = temp
            writes += 1

    free(arr)
    return writes
