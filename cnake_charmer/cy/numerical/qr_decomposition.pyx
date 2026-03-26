# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""QR decomposition via classical Gram-Schmidt (Cython-optimized).

Keywords: numerical, QR, decomposition, Gram-Schmidt, linear algebra, matrix, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin, sqrt
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200,))
def qr_decomposition(int n):
    """Compute QR decomposition of an n x n matrix via classical Gram-Schmidt."""
    cdef double *A = <double *>malloc(n * n * sizeof(double))
    if not A:
        raise MemoryError()
    cdef double *Q = <double *>malloc(n * n * sizeof(double))
    if not Q:
        free(A)
        raise MemoryError()
    cdef double *R = <double *>malloc(n * n * sizeof(double))
    if not R:
        free(A)
        free(Q)
        raise MemoryError()
    cdef double *v = <double *>malloc(n * sizeof(double))
    if not v:
        free(A)
        free(Q)
        free(R)
        raise MemoryError()

    cdef int i, j, k
    cdef double val, dot, norm, inv_norm, d, diag_norm, r00, trace

    # Build matrix A (column-major)
    for i in range(n):
        for j in range(n):
            val = sin(i * 0.7 + j * 0.3)
            if i == j:
                val = val + 2.0
            A[j * n + i] = val

    # Zero out Q and R
    for i in range(n * n):
        Q[i] = 0.0
        R[i] = 0.0

    # Classical Gram-Schmidt
    for j in range(n):
        # Copy column j of A into v
        for i in range(n):
            v[i] = A[j * n + i]

        # Subtract projections onto previous Q columns
        for k in range(j):
            dot = 0.0
            for i in range(n):
                dot += Q[k * n + i] * A[j * n + i]
            R[k * n + j] = dot
            for i in range(n):
                v[i] -= dot * Q[k * n + i]

        # R[j][j] = norm(v)
        norm = 0.0
        for i in range(n):
            norm += v[i] * v[i]
        norm = sqrt(norm)
        R[j * n + j] = norm

        # Q[:, j] = v / norm
        if norm > 1e-15:
            inv_norm = 1.0 / norm
            for i in range(n):
                Q[j * n + i] = v[i] * inv_norm

    # Compute return values
    diag_norm = 0.0
    for i in range(n):
        d = R[i * n + i]
        diag_norm += d * d
    diag_norm = sqrt(diag_norm)

    r00 = R[0]

    trace = 0.0
    for i in range(n):
        trace += R[i * n + i]

    free(A)
    free(Q)
    free(R)
    free(v)
    return (diag_norm, r00, trace)
