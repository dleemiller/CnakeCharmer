# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Longest increasing subsequence with intermediate state tracking (Cython-optimized).

Keywords: dynamic programming, LIS, longest increasing subsequence, patience sort, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(20000,))
def longest_increasing_subseq(int n):
    """Compute LIS length and intermediate state on deterministic sequence."""
    cdef long long *seq = <long long *>malloc(n * sizeof(long long))
    cdef long long *tails = <long long *>malloc(n * sizeof(long long))
    cdef int *dp = <int *>malloc(n * sizeof(int))
    cdef int i, lo, hi, mid_idx
    cdef long long val
    cdef int tails_len = 0
    cdef long long tail_array_sum = 0

    if not seq or not tails or not dp:
        free(seq); free(tails); free(dp)
        raise MemoryError()

    # Generate deterministic sequence
    cdef long long mod = <long long>n * 3
    for i in range(n):
        seq[i] = (<long long>i * 2654435761) % mod

    for i in range(n):
        val = seq[i]
        lo = 0
        hi = tails_len
        while lo < hi:
            mid_idx = (lo + hi) // 2
            if tails[mid_idx] < val:
                lo = mid_idx + 1
            else:
                hi = mid_idx
        tails[lo] = val
        dp[i] = lo + 1
        if lo == tails_len:
            tails_len += 1

    cdef int lis_length = tails_len
    cdef int dp_mid_val = dp[n // 2]
    for i in range(tails_len):
        tail_array_sum += tails[i]

    free(seq)
    free(tails)
    free(dp)
    return (lis_length, dp_mid_val, tail_array_sum)
