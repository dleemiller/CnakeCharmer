# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Counting sort on a deterministic integer array (Cython-optimized).

Keywords: algorithms, counting sort, sorting, linear time, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def counting_sort(int n):
    """Sort array using counting sort with C arrays for counts and output."""
    cdef int MAX_VAL = 1000

    cdef int *arr = <int *>malloc(n * sizeof(int))
    cdef int *counts = <int *>malloc(MAX_VAL * sizeof(int))
    cdef int *output = <int *>malloc(n * sizeof(int))

    if not arr or not counts or not output:
        if arr: free(arr)
        if counts: free(counts)
        if output: free(output)
        raise MemoryError()

    cdef int i, val, idx

    # Generate input
    for i in range(n):
        arr[i] = (i * 31 + 17) % MAX_VAL

    # Count occurrences
    memset(counts, 0, MAX_VAL * sizeof(int))
    for i in range(n):
        counts[arr[i]] += 1

    # Build sorted output directly
    idx = 0
    for val in range(MAX_VAL):
        for i in range(counts[val]):
            output[idx] = val
            idx += 1

    # Convert to Python list
    cdef list result = [output[i] for i in range(n)]

    free(arr)
    free(counts)
    free(output)
    return result
