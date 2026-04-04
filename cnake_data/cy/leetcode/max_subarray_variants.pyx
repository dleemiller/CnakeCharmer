# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Find k=3 non-overlapping maximum subarrays using DP (Cython-optimized).

Keywords: leetcode, kadane, subarray, non-overlapping, dynamic programming, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def max_subarray_variants(int n):
    """Find 3 non-overlapping subarrays of length w=n//10 with maximum total sum."""
    cdef int w, i, mid, num_subs, best_start
    cdef long long s, max_sum, total
    cdef long long *a
    cdef long long *sub_sum
    cdef int *left_best
    cdef int *right_best

    if n < 30:
        return (0, 0, 0)

    w = n // 10
    if w < 1:
        w = 1

    a = <long long *>malloc(n * sizeof(long long))
    if not a:
        raise MemoryError()

    cdef long long li, v1, v2, xv, mv
    for i in range(n):
        li = <long long>i
        v1 = li * 73856093
        v2 = li * 19349669
        xv = v1 ^ v2
        # Python-compatible modulo: always non-negative for positive divisor
        mv = xv % 201
        if mv < 0:
            mv = mv + 201
        a[i] = mv - 100

    num_subs = n - w + 1
    sub_sum = <long long *>malloc(num_subs * sizeof(long long))
    left_best = <int *>malloc(num_subs * sizeof(int))
    right_best = <int *>malloc(num_subs * sizeof(int))

    if not sub_sum or not left_best or not right_best:
        free(a)
        if sub_sum: free(sub_sum)
        if left_best: free(left_best)
        if right_best: free(right_best)
        raise MemoryError()

    s = 0
    for i in range(w):
        s = s + a[i]
    sub_sum[0] = s
    for i in range(1, num_subs):
        s = s + a[i + w - 1] - a[i - 1]
        sub_sum[i] = s

    left_best[0] = 0
    for i in range(1, num_subs):
        if sub_sum[i] > sub_sum[left_best[i - 1]]:
            left_best[i] = i
        else:
            left_best[i] = left_best[i - 1]

    right_best[num_subs - 1] = num_subs - 1
    for i in range(num_subs - 2, -1, -1):
        if sub_sum[i] >= sub_sum[right_best[i + 1]]:
            right_best[i] = i
        else:
            right_best[i] = right_best[i + 1]

    max_sum = -999999999
    best_start = 0
    for mid in range(w, num_subs - w):
        total = sub_sum[left_best[mid - w]] + sub_sum[mid] + sub_sum[right_best[mid + w]]
        if total > max_sum:
            max_sum = total
            best_start = left_best[mid - w]

    free(a)
    free(sub_sum)
    free(left_best)
    free(right_best)

    return (max_sum, best_start, 3)
