# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""LZ77 compression with match chaining (Cython-optimized).

Keywords: compression, lz77, match chain, sliding window, tokens, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(80000,))
def lz77_match_chain(int n):
    """LZ77 compression with chained match search using C arrays."""
    cdef int *s = <int *>malloc(n * sizeof(int))
    if not s:
        raise MemoryError()

    cdef int i, j, length, best_length, window_start
    cdef int max_window = 128
    cdef int min_match = 3
    cdef int num_literals = 0
    cdef int num_matches = 0
    cdef long long total_match_len = 0

    for i in range(n):
        s[i] = (i * 11 + 5) % 20

    i = 0
    while i < n:
        best_length = 0
        window_start = i - max_window if i > max_window else 0

        j = window_start
        while j < i:
            length = 0
            while i + length < n and j + length < i and s[j + length] == s[i + length]:
                length += 1
            if length > best_length:
                best_length = length
            j += 1

        if best_length >= min_match:
            num_matches += 1
            total_match_len += best_length
            i += best_length
        else:
            num_literals += 1
            i += 1

    free(s)
    return (num_literals, num_matches, int(total_match_len))
