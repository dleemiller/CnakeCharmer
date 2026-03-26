# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count all palindromic substrings using expand-around-center (Cython-optimized).

Keywords: string processing, palindrome, substring, expand center, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def palindrome_count(int n):
    """Count all palindromic substrings in a deterministic string."""
    cdef int i, center, left, right, plen
    cdef int total_count = 0
    cdef int longest = 0
    cdef long long length_sum = 0
    cdef int *text

    text = <int *>malloc(n * sizeof(int))
    if not text:
        raise MemoryError()

    for i in range(n):
        text[i] = 97 + (i * 7 + 3) % 26

    for center in range(n):
        # Odd-length palindromes
        left = center
        right = center
        while left >= 0 and right < n and text[left] == text[right]:
            plen = right - left + 1
            total_count += 1
            length_sum += plen
            if plen > longest:
                longest = plen
            left -= 1
            right += 1

        # Even-length palindromes
        left = center
        right = center + 1
        while left >= 0 and right < n and text[left] == text[right]:
            plen = right - left + 1
            total_count += 1
            length_sum += plen
            if plen > longest:
                longest = plen
            left -= 1
            right += 1

    free(text)
    return (total_count, longest, length_sum)
