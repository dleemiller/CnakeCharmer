# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute Kolmogorov-Smirnov statistic between two distributions (Cython).

Keywords: statistics, kolmogorov-smirnov, ks-test, ecdf, distribution, cython, benchmark
"""

from libc.stdlib cimport malloc, free, qsort
from cnake_charmer.benchmarks import cython_benchmark


cdef int _compare_int(const void *a, const void *b) noexcept nogil:
    return (<int *>a)[0] - (<int *>b)[0]


@cython_benchmark(syntax="cy", args=(200000,))
def kolmogorov_smirnov(int n):
    """Compute KS statistic between two deterministic sample distributions."""
    cdef int *a = <int *>malloc(n * sizeof(int))
    cdef int *b = <int *>malloc(n * sizeof(int))
    if not a or not b:
        free(a); free(b)
        raise MemoryError()

    cdef int i, j, mid_idx
    cdef double ks_stat = 0.0
    cdef int max_diff_pos = 0
    cdef double ecdf_a, ecdf_b, diff, inv_n, ecdf_mid

    for i in range(n):
        a[i] = (i * 73 + 11) % 4999
        b[i] = (i * 97 + 13) % 5003

    qsort(a, n, sizeof(int), _compare_int)
    qsort(b, n, sizeof(int), _compare_int)

    inv_n = 1.0 / n
    i = 0
    j = 0

    while i < n and j < n:
        if a[i] <= b[j]:
            i += 1
        else:
            j += 1

        ecdf_a = i * inv_n
        ecdf_b = j * inv_n
        diff = ecdf_a - ecdf_b
        if diff < 0:
            diff = -diff
        if diff > ks_stat:
            ks_stat = diff
            max_diff_pos = i + j

    while i < n:
        i += 1
        ecdf_a = i * inv_n
        ecdf_b = j * inv_n
        diff = ecdf_a - ecdf_b
        if diff < 0:
            diff = -diff
        if diff > ks_stat:
            ks_stat = diff
            max_diff_pos = i + j

    while j < n:
        j += 1
        ecdf_a = i * inv_n
        ecdf_b = j * inv_n
        diff = ecdf_a - ecdf_b
        if diff < 0:
            diff = -diff
        if diff > ks_stat:
            ks_stat = diff
            max_diff_pos = i + j

    mid_idx = n / 2
    ecdf_mid = (mid_idx + 1) * inv_n

    free(a)
    free(b)
    return (ks_stat, max_diff_pos, ecdf_mid)
