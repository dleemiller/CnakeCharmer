# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Find ECE-like segments using prefix extrema scanning (Cython).

Adapted from The Stack v2 Cython candidate:
- blob_id: ec852b83cf303a1ec1503655572dd9143cbadb35
- filename: find_ECE.pyx
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(45000, 24))
def stack_ece_segments(int length, int min_len):
    cdef int L = length + 1
    cdef int *s = <int *>malloc(L * sizeof(int))
    cdef int *r = <int *>malloc(L * sizeof(int))
    cdef int *X = <int *>malloc(L * sizeof(int))
    cdef int *Y = <int *>malloc(L * sizeof(int))
    cdef unsigned int state = 987654321
    cdef int i, j
    cdef int count = 0
    cdef int first_a = 0
    cdef int first_b = 0
    cdef long long total_span = 0

    if not s or not r or not X or not Y:
        free(s); free(r); free(X); free(Y)
        raise MemoryError()

    s[0] = 0
    for i in range(1, L):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        s[i] = 1 if (state & 7) < 5 else -4

    r[0] = 0
    for i in range(1, L):
        r[i] = r[i - 1] + s[i]

    X[0] = 0
    for i in range(1, L):
        X[i] = X[i - 1] if X[i - 1] <= r[i] else r[i]

    Y[L - 1] = r[L - 1]
    for i in range(L - 2, -1, -1):
        Y[i] = Y[i + 1] if Y[i + 1] >= r[i] else r[i]

    i = 0
    j = 0
    while j < L:
        if j == L - 1 or Y[j + 1] < X[i]:
            if j - i >= min_len:
                count += 1
                if count == 1:
                    first_a = i + 1
                    first_b = j
                total_span += (j - i)
            j += 1
            while j < L and i < L and Y[j] < X[i]:
                i += 1
        else:
            j += 1

    free(s); free(r); free(X); free(Y)
    if count > 0:
        return (count, first_a, first_b, total_span)
    return (0, 0, 0, 0)
