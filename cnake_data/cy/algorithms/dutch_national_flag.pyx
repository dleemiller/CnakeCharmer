# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Three-way partition (Dutch National Flag) on a deterministic array (Cython-optimized).

Keywords: algorithms, partition, dutch national flag, three-way, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000000,))
def dutch_national_flag(int n):
    """Partition array of 0s, 1s, 2s using Dutch National Flag algorithm with C array."""
    cdef int *arr = <int *>malloc(n * sizeof(int))
    if not arr:
        raise MemoryError()

    cdef int i, lo, mid, hi, tmp

    for i in range(n):
        arr[i] = (i * 31 + 17) % 3

    lo = 0
    mid = 0
    hi = n - 1

    while mid <= hi:
        if arr[mid] == 0:
            tmp = arr[lo]
            arr[lo] = arr[mid]
            arr[mid] = tmp
            lo += 1
            mid += 1
        elif arr[mid] == 1:
            mid += 1
        else:
            tmp = arr[mid]
            arr[mid] = arr[hi]
            arr[hi] = tmp
            hi -= 1

    cdef list result = [arr[i] for i in range(n)]
    free(arr)
    return result
