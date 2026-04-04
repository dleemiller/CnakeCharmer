# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Enriched-region segment finder using prefix sums (Cython-optimized).

Uses C-level malloc arrays for prefix sums and min/max sweeps
to find enriched contiguous segments efficiently.

Keywords: algorithms, prefix sum, segment, subsequence, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


cdef inline double double_min(double a, double b):
    return a if a <= b else b


cdef inline double double_max(double a, double b):
    return a if a >= b else b


@cython_benchmark(syntax="cy", args=(500,))
def ece_segment_finder(int n):
    """Find all ECE segments in a synthetic scored sequence."""
    # Build deterministic scored sequence
    cdef int L = n + 1
    cdef int min_ece_length = 5 if n // 20 < 5 else n // 20
    cdef int *s = <int *>malloc(L * sizeof(int))
    if not s:
        raise MemoryError()

    cdef double *r = <double *>malloc(L * sizeof(double))
    cdef double *X = <double *>malloc(L * sizeof(double))
    cdef double *Y = <double *>malloc(L * sizeof(double))
    if not r or not X or not Y:
        free(s)
        if r:
            free(r)
        if X:
            free(X)
        raise MemoryError()

    cdef int i, j

    # Fill score array
    s[0] = 0
    for i in range(1, L):
        if (i * 7) % 5 != 0:
            s[i] = 1
        else:
            s[i] = -4

    # Prefix sums
    r[0] = 0.0
    for i in range(1, L):
        r[i] = r[i - 1] + s[i]

    # Running minimum from left
    X[0] = 0.0
    for i in range(1, L):
        X[i] = double_min(X[i - 1], r[i])

    # Running maximum from right
    Y[L - 1] = r[L - 1]
    for i in range(L - 2, -1, -1):
        Y[i] = double_max(Y[i + 1], r[i])

    # Sweep to find ECE segments
    bests = []
    i = 0
    j = 0
    while j < L:
        if j == L - 1 or Y[j + 1] < X[i]:
            if j - i >= min_ece_length:
                bests.append((i + 1, j))
            j += 1
            while j < L and i < L and Y[j] < X[i]:
                i += 1
        else:
            j += 1

    free(s)
    free(r)
    free(X)
    free(Y)

    return bests
