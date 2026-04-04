# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count 2-character substrings (Cython-optimized with flat C array).

Keywords: string processing, substring counting, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def count_substrings(int n):
    """Count total 2-char substring occurrences using a flat C array of size 676."""
    cdef int *chars = <int *>malloc(n * sizeof(int))
    cdef int *counts = <int *>malloc(676 * sizeof(int))
    if not chars or not counts:
        if chars: free(chars)
        if counts: free(counts)
        raise MemoryError()

    cdef int i, c1, c2, total, unique_pairs

    # Build deterministic character array
    for i in range(n):
        chars[i] = (i * 7 + 3) % 26

    # Zero out counts
    memset(counts, 0, 676 * sizeof(int))

    # Count all 2-char pairs using flat array indexed by c1*26 + c2
    for i in range(n - 1):
        c1 = chars[i]
        c2 = chars[i + 1]
        counts[c1 * 26 + c2] += 1

    # Sum all counts and count unique pairs
    total = 0
    unique_pairs = 0
    for i in range(676):
        total += counts[i]
        if counts[i] > 0:
            unique_pairs += 1

    free(chars)
    free(counts)
    return (total, unique_pairs)
