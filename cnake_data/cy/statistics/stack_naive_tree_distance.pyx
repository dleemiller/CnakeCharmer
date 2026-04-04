# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute symmetric naive tree distance over fixed-depth context tables (Cython).

Adapted from The Stack v2 Cython candidate:
- blob_id: 5da5a7e3fb722f09faaa0bb02b89936332ffdd95
- filename: naive_parameter_sampling.pyx
"""

from libc.math cimport sqrt
from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


cdef double _asym_dist(double* A, double* B, int n_contexts, int alphabet) noexcept nogil:
    cdef int i, j, c
    cdef int clen_i, clen_j
    cdef double target, diff, sq, best
    cdef double total = 0.0

    for i in range(n_contexts):
        clen_i = 1 + (i % 5)
        for c in range(alphabet):
            target = A[i * alphabet + c]
            best = target * target
            for j in range(n_contexts):
                clen_j = 1 + (j % 5)
                if clen_i == clen_j:
                    diff = B[j * alphabet + c] - target
                    sq = diff * diff
                    if sq < best:
                        best = sq
            total += best

    return sqrt(total / (n_contexts * alphabet))


@cython_benchmark(syntax="cy", args=(700, 4))
def stack_naive_tree_distance(int n_contexts, int alphabet):
    cdef unsigned int state = 13579
    cdef double *left = <double *>malloc(n_contexts * alphabet * sizeof(double))
    cdef double *right = <double *>malloc(n_contexts * alphabet * sizeof(double))
    cdef int i, c
    cdef double s1, s2, v1, v2
    cdef double d1, d2, sym

    if not left or not right:
        free(left); free(right)
        raise MemoryError()

    for i in range(n_contexts):
        s1 = 0.0
        s2 = 0.0
        for c in range(alphabet):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            v1 = 1.0 + ((state >> 8) % 100)
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            v2 = 1.0 + ((state >> 8) % 100)
            left[i * alphabet + c] = v1
            right[i * alphabet + c] = v2
            s1 += v1
            s2 += v2
        for c in range(alphabet):
            left[i * alphabet + c] /= s1
            right[i * alphabet + c] /= s2

    d1 = _asym_dist(left, right, n_contexts, alphabet)
    d2 = _asym_dist(right, left, n_contexts, alphabet)
    sym = 0.5 * (d1 + d2)

    free(left)
    free(right)
    return (<int>(d1 * 1000000), <int>(d2 * 1000000), <int>(sym * 1000000), n_contexts)
