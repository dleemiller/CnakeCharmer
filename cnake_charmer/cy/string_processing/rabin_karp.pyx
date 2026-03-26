# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count pattern occurrences using Rabin-Karp rolling hash (Cython-optimized).

Keywords: string processing, rabin karp, rolling hash, pattern matching, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def rabin_karp(int n):
    """Count occurrences of a 5-char pattern in a deterministic text using Rabin-Karp.

    Args:
        n: Length of the text.

    Returns:
        Number of pattern occurrences.
    """
    cdef int i, j
    cdef int pat_len = 5
    cdef long long base = 256
    cdef long long mod = 1000000007
    cdef long long h, pat_hash, win_hash
    cdef int count = 0
    cdef int match
    cdef int *text
    cdef int pattern[5]

    if n < 5:
        return 0

    text = <int *>malloc(n * sizeof(int))
    if not text:
        raise MemoryError()

    # Build text
    for i in range(n):
        text[i] = 65 + (i * 7 + 3) % 26

    # Store pattern
    for i in range(pat_len):
        pattern[i] = text[i]

    # Compute base^(pat_len-1) % mod
    h = 1
    for i in range(pat_len - 1):
        h = (h * base) % mod

    # Hash the pattern and first window
    pat_hash = 0
    win_hash = 0
    for i in range(pat_len):
        pat_hash = (pat_hash * base + pattern[i]) % mod
        win_hash = (win_hash * base + text[i]) % mod

    for i in range(n - pat_len + 1):
        if win_hash == pat_hash:
            match = 1
            for j in range(pat_len):
                if text[i + j] != pattern[j]:
                    match = 0
                    break
            if match:
                count += 1
        # Slide the window
        if i < n - pat_len:
            win_hash = ((win_hash - text[i] * h) * base + text[i + pat_len]) % mod
            if win_hash < 0:
                win_hash += mod

    free(text)
    return count
