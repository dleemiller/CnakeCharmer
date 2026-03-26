# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count pattern occurrences in text using KMP algorithm (Cython-optimized).

Keywords: algorithms, string matching, KMP, pattern search, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def kmp_search(int n):
    """Count pattern occurrences using KMP with C char arrays."""
    cdef int m = 10  # pattern length

    cdef char *text = <char *>malloc(n * sizeof(char))
    cdef char *pattern = <char *>malloc(m * sizeof(char))
    cdef int *fail = <int *>malloc(m * sizeof(int))

    if not text or not pattern or not fail:
        if text: free(text)
        if pattern: free(pattern)
        if fail: free(fail)
        raise MemoryError()

    cdef int i, k
    cdef int count = 0
    cdef int last_match_pos = -1

    # Build text and pattern
    for i in range(n):
        text[i] = <char>(65 + (i * 7 + 3) % 26)
    for i in range(m):
        pattern[i] = <char>(65 + (i * 7 + 3) % 26)

    # Build KMP failure function
    fail[0] = 0
    k = 0
    for i in range(1, m):
        while k > 0 and pattern[k] != pattern[i]:
            k = fail[k - 1]
        if pattern[k] == pattern[i]:
            k += 1
        fail[i] = k

    # Search
    k = 0
    for i in range(n):
        while k > 0 and pattern[k] != text[i]:
            k = fail[k - 1]
        if pattern[k] == text[i]:
            k += 1
        if k == m:
            count += 1
            last_match_pos = i - m + 1
            k = fail[k - 1]

    free(text)
    free(pattern)
    free(fail)
    return (count, last_match_pos)
