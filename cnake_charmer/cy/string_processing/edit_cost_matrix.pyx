# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Edit distance cost matrix (Cython-optimized).

Computes the full DP matrix for transforming string s into string t,
using a flat C array with malloc/free.

Keywords: string processing, edit distance, cost matrix, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark
from libc.stdlib cimport malloc, free


@cython_benchmark(syntax="cy", args=(500, 2))
def edit_cost_matrix(int n, int substitution_cost):
    """Compute the full edit-distance cost matrix using flat C arrays."""
    # Generate deterministic strings
    cdef int s_len = n
    cdef int t_len = n + n // 5
    cdef int i, j, cost, del_cost, ins_cost, sub_cost
    cdef int m = s_len
    cdef int p = t_len

    cdef unsigned char *s = <unsigned char *>malloc(s_len)
    cdef unsigned char *t = <unsigned char *>malloc(t_len)
    cdef int *dp = <int *>malloc((m + 1) * (p + 1) * sizeof(int))

    if s == NULL or t == NULL or dp == NULL:
        if s != NULL:
            free(s)
        if t != NULL:
            free(t)
        if dp != NULL:
            free(dp)
        raise MemoryError("Failed to allocate arrays")

    # Fill deterministic strings
    for i in range(s_len):
        s[i] = (i * 7 + 3) % 26 + 97
    for i in range(t_len):
        t[i] = (i * 11 + 5) % 26 + 97

    # Initialize base cases
    for i in range(m + 1):
        dp[i * (p + 1)] = i
    for j in range(p + 1):
        dp[j] = j

    # Fill cost matrix
    for i in range(1, m + 1):
        for j in range(1, p + 1):
            if s[i - 1] == t[j - 1]:
                dp[i * (p + 1) + j] = dp[(i - 1) * (p + 1) + (j - 1)]
            else:
                del_cost = dp[(i - 1) * (p + 1) + j] + 1
                ins_cost = dp[i * (p + 1) + (j - 1)] + 1
                sub_cost = dp[(i - 1) * (p + 1) + (j - 1)] + substitution_cost
                cost = del_cost
                if ins_cost < cost:
                    cost = ins_cost
                if sub_cost < cost:
                    cost = sub_cost
                dp[i * (p + 1) + j] = cost

    cdef int edit_dist = dp[m * (p + 1) + p]
    cdef int cost_at_mid = dp[(m // 2) * (p + 1) + (p // 2)]
    cdef long long checksum = 0
    for j in range(p + 1):
        checksum += dp[m * (p + 1) + j]

    free(s)
    free(t)
    free(dp)

    return (edit_dist, cost_at_mid, <int>checksum)
