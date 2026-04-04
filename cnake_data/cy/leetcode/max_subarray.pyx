# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Maximum subarray sum using Kadane's algorithm.

Keywords: leetcode, max subarray, kadane, dynamic programming, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000000,))
def max_subarray(int n):
    """Find the maximum subarray sum using Kadane's algorithm."""
    cdef long long best, current, val
    cdef int i

    best = ((0 * 17 + 5) % 201) - 100
    current = best

    for i in range(1, n):
        val = ((i * 17 + 5) % 201) - 100
        if current + val > val:
            current = current + val
        else:
            current = val
        if current > best:
            best = current

    return int(best)
