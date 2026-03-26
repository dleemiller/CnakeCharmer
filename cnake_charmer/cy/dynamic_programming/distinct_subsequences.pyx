# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Count distinct subsequences of one string in another using DP (Cython-optimized).

Keywords: dynamic programming, distinct subsequences, string, counting, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1500,))
def distinct_subsequences(int n):
    """Count distinct subsequences using two rolling C arrays."""
    cdef long long MOD = 1000000007
    cdef int len_s = n
    cdef int len_t = n / 5
    cdef int i, j, mid_s, mid_t
    cdef long long mid_val = 0
    cdef int *s = <int *>malloc(len_s * sizeof(int))
    cdef int *t = <int *>malloc(len_t * sizeof(int))
    cdef long long *dp_prev = <long long *>malloc((len_t + 1) * sizeof(long long))
    cdef long long *dp_curr = <long long *>malloc((len_t + 1) * sizeof(long long))
    cdef long long *tmp

    if s == NULL or t == NULL or dp_prev == NULL or dp_curr == NULL:
        if s != NULL:
            free(s)
        if t != NULL:
            free(t)
        if dp_prev != NULL:
            free(dp_prev)
        if dp_curr != NULL:
            free(dp_curr)
        raise MemoryError("Failed to allocate arrays")

    # Build strings as int arrays
    for i in range(len_s):
        s[i] = (i * 7 + 3) % 3
    # t[j] = s[j*5] (every 5th char of s)
    for i in range(len_t):
        t[i] = s[i * 5]

    # Initialize dp_prev
    dp_prev[0] = 1
    for j in range(1, len_t + 1):
        dp_prev[j] = 0

    mid_s = len_s / 2
    mid_t = len_t / 2

    for i in range(1, len_s + 1):
        dp_curr[0] = 1
        for j in range(1, len_t + 1):
            dp_curr[j] = dp_prev[j] % MOD
            if s[i - 1] == t[j - 1]:
                dp_curr[j] = (dp_curr[j] + dp_prev[j - 1]) % MOD
        if i == mid_s:
            mid_val = dp_curr[mid_t]
        # Swap pointers
        tmp = dp_prev
        dp_prev = dp_curr
        dp_curr = tmp

    cdef long long result = dp_prev[len_t]

    free(s)
    free(t)
    free(dp_prev)
    free(dp_curr)
    return (result, mid_val)
