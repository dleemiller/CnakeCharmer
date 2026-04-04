# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Rabin-Karp string matching on deterministic text and pattern (Cython-optimized).

Keywords: algorithms, rabin-karp, string matching, hashing, search, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def rabin_karp_search(int n):
    """Search for pattern occurrences using Rabin-Karp with C arrays."""
    cdef int *text = <int *>malloc(n * sizeof(int))
    if not text:
        raise MemoryError()

    cdef int pat_len = 7
    cdef int pattern[7]
    cdef int i, j
    cdef long long base = 256
    cdef long long mod = 1000000007
    cdef long long pat_hash = 0
    cdef long long win_hash = 0
    cdef long long h = 1
    cdef int match_count = 0
    cdef int first_match = -1
    cdef int last_match = -1
    cdef int match

    # Generate deterministic text
    for i in range(n):
        text[i] = 97 + ((i * 7 + 13) % 26)

    # Generate pattern
    for i in range(pat_len):
        pattern[i] = 97 + ((i * 7 + 13) % 26)

    # Compute h = base^(pat_len-1) mod mod
    for i in range(pat_len - 1):
        h = (h * base) % mod

    # Compute initial hashes
    for i in range(pat_len):
        pat_hash = (pat_hash * base + pattern[i]) % mod
        win_hash = (win_hash * base + text[i]) % mod

    for i in range(n - pat_len + 1):
        if win_hash == pat_hash:
            # Verify match
            match = 1
            for j in range(pat_len):
                if text[i + j] != pattern[j]:
                    match = 0
                    break
            if match:
                match_count += 1
                if first_match == -1:
                    first_match = i
                last_match = i

        # Roll hash forward
        if i < n - pat_len:
            win_hash = (base * (win_hash - text[i] * h) + text[i + pat_len]) % mod
            if win_hash < 0:
                win_hash += mod

    free(text)
    return (match_count, first_match, last_match)
