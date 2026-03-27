# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Spearman rank correlation with detailed statistics (Cython).

Keywords: statistics, spearman, rank, correlation, d-squared, cython, benchmark
"""

from libc.stdlib cimport malloc, free, qsort
from libc.math cimport sqrt
from cnake_charmer.benchmarks import cython_benchmark


cdef struct IndexedVal:
    int val
    int idx


cdef int _compare_indexed(const void *a, const void *b) noexcept nogil:
    cdef IndexedVal *ia = <IndexedVal *>a
    cdef IndexedVal *ib = <IndexedVal *>b
    return ia.val - ib.val


@cython_benchmark(syntax="cy", args=(100000,))
def spearman_rank(int n):
    """Compute Spearman rank correlation with additional statistics."""
    cdef int *x = <int *>malloc(n * sizeof(int))
    cdef int *y = <int *>malloc(n * sizeof(int))
    cdef double *rx = <double *>malloc(n * sizeof(double))
    cdef double *ry = <double *>malloc(n * sizeof(double))
    cdef IndexedVal *indexed = <IndexedVal *>malloc(n * sizeof(IndexedVal))
    if not x or not y or not rx or not ry or not indexed:
        free(x); free(y); free(rx); free(ry); free(indexed)
        raise MemoryError()

    cdef int i, j
    cdef double avg_rank, mean_rx, mean_ry, mean_rank
    cdef double cov, var_x, var_y, dx, dy, d, sum_d_sq
    cdef double correlation

    # Generate data
    for i in range(n):
        x[i] = (i * 11 + 5) % 1009
        y[i] = (i * 19 + 3) % 1013

    # Rank x
    for i in range(n):
        indexed[i].val = x[i]
        indexed[i].idx = i
    qsort(indexed, n, sizeof(IndexedVal), _compare_indexed)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and indexed[j + 1].val == indexed[j].val:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        while i <= j:
            rx[indexed[i].idx] = avg_rank
            i += 1

    # Rank y
    for i in range(n):
        indexed[i].val = y[i]
        indexed[i].idx = i
    qsort(indexed, n, sizeof(IndexedVal), _compare_indexed)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and indexed[j + 1].val == indexed[j].val:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        while i <= j:
            ry[indexed[i].idx] = avg_rank
            i += 1

    # sum_d_sq and mean_rank
    sum_d_sq = 0.0
    mean_rank = 0.0
    for i in range(n):
        d = rx[i] - ry[i]
        sum_d_sq += d * d
        mean_rank += rx[i]
    mean_rank = mean_rank / n

    # Pearson correlation on ranks
    mean_rx = 0.0
    mean_ry = 0.0
    for i in range(n):
        mean_rx += rx[i]
        mean_ry += ry[i]
    mean_rx /= n
    mean_ry /= n

    cov = 0.0
    var_x = 0.0
    var_y = 0.0
    for i in range(n):
        dx = rx[i] - mean_rx
        dy = ry[i] - mean_ry
        cov += dx * dy
        var_x += dx * dx
        var_y += dy * dy

    free(x); free(y); free(rx); free(ry); free(indexed)

    if var_x == 0.0 or var_y == 0.0:
        correlation = 0.0
    else:
        correlation = cov / sqrt(var_x * var_y)

    return (correlation, sum_d_sq, mean_rank)
