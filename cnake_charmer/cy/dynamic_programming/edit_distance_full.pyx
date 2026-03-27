# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute edit distance with full alignment traceback (Cython-optimized).

Keywords: dynamic programming, edit distance, levenshtein, alignment, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1500,))
def edit_distance_full(int n):
    """Compute edit distance between two deterministic strings of length n."""
    cdef int i, j, mid, stride, num_subs
    cdef int sub, ins, dlt, best, diag, up, left, distance, dp_mid_mid

    cdef int *a = <int *>malloc(n * sizeof(int))
    cdef int *b = <int *>malloc(n * sizeof(int))
    stride = n + 1
    cdef int *dp = <int *>malloc((n + 1) * stride * sizeof(int))

    if not a or not b or not dp:
        if a: free(a)
        if b: free(b)
        if dp: free(dp)
        raise MemoryError()

    for i in range(n):
        a[i] = (i * 7 + 3) % 26
        b[i] = (i * 13 + 5) % 26

    for i in range(n + 1):
        dp[i * stride + 0] = i
    for j in range(n + 1):
        dp[0 * stride + j] = j

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i * stride + j] = dp[(i - 1) * stride + (j - 1)]
            else:
                sub = dp[(i - 1) * stride + (j - 1)] + 1
                ins = dp[i * stride + (j - 1)] + 1
                dlt = dp[(i - 1) * stride + j] + 1
                best = sub
                if ins < best:
                    best = ins
                if dlt < best:
                    best = dlt
                dp[i * stride + j] = best

    distance = dp[n * stride + n]

    mid = n // 2
    dp_mid_mid = dp[mid * stride + mid]

    # Traceback to count substitutions
    num_subs = 0
    i = n
    j = n
    while i > 0 and j > 0:
        diag = dp[(i - 1) * stride + (j - 1)]
        up = dp[(i - 1) * stride + j]
        left = dp[i * stride + (j - 1)]

        if a[i - 1] == b[j - 1]:
            i = i - 1
            j = j - 1
        elif diag <= up and diag <= left:
            num_subs = num_subs + 1
            i = i - 1
            j = j - 1
        elif up <= left:
            i = i - 1
        else:
            j = j - 1

    free(a)
    free(b)
    free(dp)

    return (distance, dp_mid_mid, num_subs)
