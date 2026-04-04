# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Longest increasing subsequence (Cython-optimized).

Keywords: dynamic programming, longest increasing subsequence, LIS, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark
from libc.stdlib cimport malloc, free


@cython_benchmark(syntax="cy", args=(2000,))
def longest_increasing_subsequence(int n):
    """Compute LIS length using typed C arrays and binary search."""
    cdef int *seq = <int *>malloc(n * sizeof(int))
    cdef int *tails = <int *>malloc(n * sizeof(int))
    cdef int tails_len = 0
    cdef int i, val, lo, hi, mid
    cdef int result, last_tail

    if seq == NULL or tails == NULL:
        if seq != NULL:
            free(seq)
        if tails != NULL:
            free(tails)
        raise MemoryError("Failed to allocate arrays")

    for i in range(n):
        seq[i] = (i * 7) % n

    for i in range(n):
        val = seq[i]
        lo = 0
        hi = tails_len
        while lo < hi:
            mid = (lo + hi) / 2
            if tails[mid] < val:
                lo = mid + 1
            else:
                hi = mid
        if lo == tails_len:
            tails[tails_len] = val
            tails_len += 1
        else:
            tails[lo] = val

    result = tails_len
    last_tail = tails[tails_len - 1] if tails_len > 0 else 0
    free(seq)
    free(tails)
    return (result, last_tail)
