# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Permutation flow shop mean completion time — Cython implementation."""

from libc.math cimport sin
from libc.stdlib cimport free, malloc

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(40, 8))
def flow_shop_makespan(int n, int m):
    """Compute mean job completion time for a permutation flow shop.

    Generates deterministic processing times from (n, m) via sin,
    then evaluates the identity permutation [1, 2, ..., n].

    Args:
        n: Number of jobs.
        m: Number of machines.

    Returns:
        Mean completion time (float).
    """
    cdef int i, j, idx, prev, c
    cdef double total, val
    cdef int *orders = <int *>malloc(n * m * sizeof(int))
    cdef int *mat = <int *>malloc(n * m * sizeof(int))
    cdef int *sol = <int *>malloc(n * sizeof(int))
    if not orders or not mat or not sol:
        free(orders); free(mat); free(sol)
        raise MemoryError()

    # Generate deterministic processing times in [1, 20]
    for i in range(n):
        for j in range(m):
            orders[i * m + j] = <int>(10.0 + 9.0 * sin(i * 1.3 + j * 0.7)) + 1

    # Identity permutation (1-indexed)
    for i in range(n):
        sol[i] = i + 1

    # First job
    idx = sol[0] - 1
    c = 0
    for j in range(m):
        c += orders[idx * m + j]
        mat[idx * m + j] = c
    total = <double>mat[idx * m + m - 1]

    # Remaining jobs
    for i in range(1, n):
        idx = sol[i] - 1
        prev = sol[i - 1] - 1
        mat[idx * m] = mat[prev * m] + orders[idx * m]
        for j in range(1, m):
            val = mat[prev * m + j]
            if mat[idx * m + j - 1] > val:
                val = mat[idx * m + j - 1]
            mat[idx * m + j] = <int>val + orders[idx * m + j]
        total += mat[idx * m + m - 1]

    free(orders)
    free(mat)
    free(sol)
    return total / n
