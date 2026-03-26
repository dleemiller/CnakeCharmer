# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Sum of maximum subarray sums using Kadane's algorithm (Cython-optimized).

Keywords: dynamic programming, kadane, max subarray, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def max_subarray_sum(int n):
    """Find sum of max subarray sums using C arrays and Kadane's algorithm."""
    cdef int chunk_size = 100
    cdef int k = n / chunk_size
    cdef long long total = 0
    cdef int chunk, i, offset, val
    cdef int max_ending_here, max_so_far
    cdef int max_single = -101
    cdef int *v = <int *>malloc(chunk_size * sizeof(int))

    if v == NULL:
        raise MemoryError("Failed to allocate array")

    for chunk in range(k):
        offset = chunk * chunk_size

        # Generate chunk values
        for i in range(chunk_size):
            v[i] = ((offset + i) * 17 + 5) % 201 - 100

        # Kadane's algorithm
        max_ending_here = v[0]
        max_so_far = v[0]
        for i in range(1, chunk_size):
            if max_ending_here + v[i] > v[i]:
                max_ending_here = max_ending_here + v[i]
            else:
                max_ending_here = v[i]
            if max_ending_here > max_so_far:
                max_so_far = max_ending_here

        total += max_so_far
        if max_so_far > max_single:
            max_single = max_so_far

    free(v)
    return (total, max_single)
