# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Scaled dot-product attention scores.

Keywords: attention, transformer, dot product, neural network, self-attention, cython
"""

from libc.math cimport sin, cos, sqrt
from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500,))
def attention_scores(int n):
    """Compute scaled dot-product attention scores Q*K^T/sqrt(d)."""
    cdef int d = 64
    cdef double inv_sqrt_d = 1.0 / sqrt(<double>d)
    cdef int i, j, k
    cdef double dot, total

    # Precompute Q and K matrices
    cdef double *q_mat = <double *>malloc(n * d * sizeof(double))
    cdef double *k_mat = <double *>malloc(n * d * sizeof(double))
    if not q_mat or not k_mat:
        if q_mat: free(q_mat)
        if k_mat: free(k_mat)
        raise MemoryError()

    for i in range(n):
        for k in range(d):
            q_mat[i * d + k] = sin(i * 0.1 + k * 0.01)
            k_mat[i * d + k] = cos(i * 0.1 + k * 0.01)

    total = 0.0
    for i in range(n):
        for j in range(n):
            dot = 0.0
            for k in range(d):
                dot += q_mat[i * d + k] * k_mat[j * d + k]
            total += dot * inv_sqrt_d

    free(q_mat)
    free(k_mat)
    return total
