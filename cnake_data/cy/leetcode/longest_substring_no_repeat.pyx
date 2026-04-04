# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Longest substring without repeating characters (Cython-optimized).

Keywords: leetcode, sliding window, substring, no repeat, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def longest_substring_no_repeat(int n):
    """Find the longest substring without repeating characters."""
    cdef int *codes = <int *>malloc(n * sizeof(int))
    cdef int *last_seen = <int *>malloc(128 * sizeof(int))
    if not codes or not last_seen:
        if codes:
            free(codes)
        if last_seen:
            free(last_seen)
        raise MemoryError()

    cdef int i, c, left, right, window_len
    cdef int max_len = 0
    cdef int max_start = 0
    cdef int adjustments = 0

    # Generate deterministic character codes in range [0, 128)
    for i in range(n):
        codes[i] = (<unsigned int>i * <unsigned int>2654435761) % 128

    # Initialize last_seen to -1
    for i in range(128):
        last_seen[i] = -1

    left = 0
    for right in range(n):
        c = codes[right]
        if last_seen[c] >= left:
            left = last_seen[c] + 1
            adjustments += 1
        last_seen[c] = right
        window_len = right - left + 1
        if window_len > max_len:
            max_len = window_len
            max_start = left

    free(codes)
    free(last_seen)
    return (max_len, max_start, adjustments)
