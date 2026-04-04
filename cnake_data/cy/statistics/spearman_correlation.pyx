# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Spearman rank correlation coefficient (Cython-optimized).

Keywords: statistics, spearman, correlation, ranking, cython, benchmark
"""

from libc.stdlib cimport malloc, free, qsort
from libc.math cimport sqrt
from cnake_data.benchmarks import cython_benchmark


cdef struct IndexedVal:
    int val
    int idx


cdef int _compare_indexed(const void *a, const void *b) noexcept nogil:
    cdef IndexedVal *ia = <IndexedVal *>a
    cdef IndexedVal *ib = <IndexedVal *>b
    return ia.val - ib.val


@cython_benchmark(syntax="cy", args=(100000,))
def spearman_correlation(int n):
    """Compute Spearman rank correlation of two sequences of length n."""
    cdef int i, j
    cdef double avg_rank, mean_rx, mean_ry
    cdef double cov, var_x, var_y, dx, dy

    cdef int *x = <int *>malloc(n * sizeof(int))
    cdef int *y = <int *>malloc(n * sizeof(int))
    cdef double *rx = <double *>malloc(n * sizeof(double))
    cdef double *ry = <double *>malloc(n * sizeof(double))
    cdef IndexedVal *indexed = <IndexedVal *>malloc(n * sizeof(IndexedVal))
    if not x or not y or not rx or not ry or not indexed:
        free(x); free(y); free(rx); free(ry); free(indexed)
        raise MemoryError()

    # Generate data
    for i in range(n):
        x[i] = (i * 7 + 3) % 1000
        y[i] = (i * 13 + 7) % 1000

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
        return 0.0
    return cov / sqrt(var_x * var_y)
