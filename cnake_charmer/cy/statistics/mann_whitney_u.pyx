# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute Mann-Whitney U statistic for two deterministic samples (Cython).

Keywords: statistics, mann-whitney, u-test, nonparametric, rank sum, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(3000,))
def mann_whitney_u(int n):
    """Compute Mann-Whitney U statistic for two deterministic samples."""
    cdef int *a = <int *>malloc(n * sizeof(int))
    cdef int *b = <int *>malloc(n * sizeof(int))
    if not a or not b:
        free(a); free(b)
        raise MemoryError()

    cdef int i, j, ai, bj
    cdef long long n_greater = 0
    cdef long long tied_count = 0
    cdef double u_stat

    for i in range(n):
        a[i] = (i * 17 + 3) % 2003
        b[i] = (i * 31 + 7) % 1999

    for i in range(n):
        ai = a[i]
        for j in range(n):
            bj = b[j]
            if ai > bj:
                n_greater += 1
            elif ai == bj:
                tied_count += 1

    u_stat = n_greater + tied_count * 0.5

    free(a)
    free(b)
    return (u_stat, n_greater, tied_count)
