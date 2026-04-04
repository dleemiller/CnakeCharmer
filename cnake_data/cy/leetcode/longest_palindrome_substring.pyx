# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Find the longest palindromic substring (Cython-optimized).

Keywords: leetcode, palindrome, substring, expand around center, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(20000,))
def longest_palindrome_substring(int n):
    """Find longest palindromic substring using expand-around-center with C arrays."""
    cdef int *s = <int *>malloc(n * sizeof(int))
    if not s:
        raise MemoryError()

    cdef int i, center, left, right, length
    cdef int max_len = 1
    cdef int best_start = 0
    cdef int center_idx, center_char

    for i in range(n):
        s[i] = (i * i + 3 * i + 7) % 8

    for center in range(n):
        # Odd-length palindromes
        left = center
        right = center
        while left >= 0 and right < n and s[left] == s[right]:
            length = right - left + 1
            if length > max_len:
                max_len = length
                best_start = left
            left -= 1
            right += 1

        # Even-length palindromes
        left = center
        right = center + 1
        while left >= 0 and right < n and s[left] == s[right]:
            length = right - left + 1
            if length > max_len:
                max_len = length
                best_start = left
            left -= 1
            right += 1

    center_idx = best_start + max_len // 2
    center_char = s[center_idx]

    free(s)
    return (max_len, best_start, center_char)
