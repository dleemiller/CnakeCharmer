# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Shell sort with Knuth gap sequence (Cython-optimized).

Keywords: sorting, shell sort, knuth gaps, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def shell_sort(int n):
    """Sort a deterministic array using Shell sort with Knuth gaps."""
    cdef int *arr = <int *>malloc(n * sizeof(int))
    if not arr:
        raise MemoryError()

    cdef int i, j, gap, temp
    cdef int num_gaps, g

    # Generate deterministic array
    for i in range(n):
        arr[i] = (i * 31 + 17) % n

    # Compute Knuth gaps
    cdef int *gaps = <int *>malloc(50 * sizeof(int))
    if not gaps:
        free(arr)
        raise MemoryError()

    num_gaps = 0
    gap = 1
    while gap < n:
        gaps[num_gaps] = gap
        num_gaps += 1
        gap = 3 * gap + 1

    # Shell sort with Knuth gaps (largest to smallest)
    for g in range(num_gaps - 1, -1, -1):
        gap = gaps[g]
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp

    cdef list result = [arr[i] for i in range(n)]
    free(arr)
    free(gaps)
    return result
