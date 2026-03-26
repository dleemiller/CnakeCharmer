# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Radix sort (base 10, LSD) on a deterministic integer array (Cython-optimized).

Keywords: algorithms, radix sort, sorting, LSD, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def radix_sort(int n):
    """Radix sort using C arrays for buckets and counting sort per digit."""
    cdef int *arr = <int *>malloc(n * sizeof(int))
    cdef int *output = <int *>malloc(n * sizeof(int))
    cdef int count[10]
    if not arr or not output:
        if arr: free(arr)
        if output: free(output)
        raise MemoryError()

    cdef int i, digit, exp_val
    cdef int max_val = 99999

    for i in range(n):
        arr[i] = (i * 31 + 17) % 100000

    exp_val = 1
    while exp_val <= max_val:
        # Reset counts
        memset(count, 0, 10 * sizeof(int))

        # Count occurrences of each digit
        for i in range(n):
            digit = (arr[i] / exp_val) % 10
            count[digit] += 1

        # Cumulative count
        for i in range(1, 10):
            count[i] += count[i - 1]

        # Build output (stable, iterate in reverse)
        for i in range(n - 1, -1, -1):
            digit = (arr[i] / exp_val) % 10
            count[digit] -= 1
            output[count[digit]] = arr[i]

        # Copy output back to arr
        for i in range(n):
            arr[i] = output[i]

        exp_val *= 10

    result = [arr[i] for i in range(n)]
    free(arr)
    free(output)
    return result
