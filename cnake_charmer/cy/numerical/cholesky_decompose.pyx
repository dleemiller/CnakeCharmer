# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Cholesky decomposition (Cython-optimized).

Keywords: numerical, cholesky, decomposition, linear algebra, matrix, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, fabs
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500,))
def cholesky_decompose(int n):
    """Compute Cholesky decomposition and return sum of diagonal of L."""
    cdef double *A = <double *>malloc(n * n * sizeof(double))
    if not A:
        raise MemoryError()

    cdef double *L = <double *>malloc(n * n * sizeof(double))
    if not L:
        free(A)
        raise MemoryError()

    cdef int i, j, k
    cdef double s, diag_sum
    cdef int diff

    # Build positive-definite matrix
    for i in range(n):
        for j in range(n):
            diff = i - j
            if diff < 0:
                diff = -diff
            A[i * n + j] = 1.0 / (diff + 1)

    # Zero out L
    for i in range(n * n):
        L[i] = 0.0

    # Cholesky decomposition
    for i in range(n):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s += L[i * n + k] * L[j * n + k]

            if i == j:
                L[i * n + j] = sqrt(A[i * n + i] - s)
            else:
                L[i * n + j] = (A[i * n + j] - s) / L[j * n + j]

    # Sum diagonal of L
    diag_sum = 0.0
    for i in range(n):
        diag_sum += L[i * n + i]

    free(A)
    free(L)
    return diag_sum
