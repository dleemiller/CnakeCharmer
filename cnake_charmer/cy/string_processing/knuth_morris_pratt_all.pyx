# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""KMP pattern matching returning all match positions (Cython-optimized).

Keywords: string processing, KMP, Knuth-Morris-Pratt, pattern matching, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2000000,))
def knuth_morris_pratt_all(int n):
    """Find all occurrences of a pattern in deterministic text using KMP."""
    cdef int pat_len = 6
    cdef int i, j, k
    cdef int match_count = 0
    cdef int last_pos = -1
    cdef long long fail_checksum = 0
    cdef int *text
    cdef int pattern[6]
    cdef int failure[6]

    if n < pat_len:
        return (0, -1, 0)

    text = <int *>malloc(n * sizeof(int))
    if not text:
        raise MemoryError()

    # Build text
    for i in range(n):
        text[i] = 97 + (i * 7 + 3) % 26

    # Store pattern
    for i in range(pat_len):
        pattern[i] = text[i]

    # Build failure table
    for i in range(pat_len):
        failure[i] = 0

    k = 0
    for i in range(1, pat_len):
        while k > 0 and pattern[k] != pattern[i]:
            k = failure[k - 1]
        if pattern[k] == pattern[i]:
            k += 1
        failure[i] = k

    # Compute failure table checksum
    for i in range(pat_len):
        fail_checksum += <long long>failure[i] * <long long>(i + 1)

    # KMP search
    j = 0
    for i in range(n):
        while j > 0 and pattern[j] != text[i]:
            j = failure[j - 1]
        if pattern[j] == text[i]:
            j += 1
        if j == pat_len:
            match_count += 1
            last_pos = i - pat_len + 1
            j = failure[j - 1]

    free(text)
    return (match_count, last_pos, fail_checksum)
