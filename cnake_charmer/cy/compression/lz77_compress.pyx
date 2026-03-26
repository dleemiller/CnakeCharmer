# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""LZ77-style compression match counting (Cython-optimized).

Keywords: compression, lz77, string matching, sliding window, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def lz77_compress(int n):
    """Count LZ77-style match pairs in a deterministic string."""
    cdef char *s = <char *>malloc(n * sizeof(char))
    if not s:
        raise MemoryError()

    cdef int i, j, length, best_length, window_start
    cdef int match_count = 0
    cdef int max_window = 100

    # Generate deterministic string
    for i in range(n):
        s[i] = 65 + (i * 7 + 3) % 26

    i = 0
    while i < n:
        best_length = 0
        window_start = i - max_window
        if window_start < 0:
            window_start = 0

        for j in range(window_start, i):
            length = 0
            while i + length < n and s[j + length] == s[i + length] and j + length < i:
                length += 1
            if length > best_length:
                best_length = length

        if best_length >= 2:
            match_count += 1
            i += best_length
        else:
            i += 1

    free(s)
    return match_count
