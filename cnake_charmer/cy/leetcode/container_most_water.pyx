# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Container with most water (Cython-optimized two pointer approach).

Keywords: leetcode, container most water, two pointer, greedy, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(3000000,))
def container_most_water(int n):
    """Find the container with most water using two pointers."""
    cdef int *heights = <int *>malloc(n * sizeof(int))
    if not heights:
        raise MemoryError()

    cdef int i, left, right, h_left, h_right
    cdef int best_left = 0
    cdef int best_right = 0
    cdef int moves = 0
    cdef long long area
    cdef long long max_area = 0

    # Generate deterministic heights
    for i in range(n):
        heights[i] = ((<unsigned int>i * <unsigned int>2654435761) % 1000000) + 1

    left = 0
    right = n - 1

    while left < right:
        h_left = heights[left]
        h_right = heights[right]
        if h_left < h_right:
            area = <long long>h_left * <long long>(right - left)
        else:
            area = <long long>h_right * <long long>(right - left)

        if area > max_area:
            max_area = area
            best_left = left
            best_right = right

        if h_left <= h_right:
            left += 1
        else:
            right -= 1
        moves += 1

    free(heights)
    return (int(max_area), best_left, best_right, moves)
