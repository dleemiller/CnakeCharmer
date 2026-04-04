# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""LU decomposition (Cython-optimized).

Keywords: numerical, lu decomposition, linear algebra, matrix, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(300,))
def lu_decomposition(int n):
    """Compute LU decomposition and return sum of diagonal of U."""
    cdef double *A = <double *>malloc(n * n * sizeof(double))
    if not A:
        raise MemoryError()

    cdef int i, j, k
    cdef double diag_sum

    # Build matrix
    for i in range(n):
        for j in range(n):
            A[i * n + j] = <double>((i * j + i + j + 1) % 100)

    # LU decomposition in-place (Doolittle algorithm)
    for k in range(n):
        for i in range(k + 1, n):
            if A[k * n + k] == 0.0:
                continue
            A[i * n + k] = A[i * n + k] / A[k * n + k]
            for j in range(k + 1, n):
                A[i * n + j] -= A[i * n + k] * A[k * n + j]

    # Sum diagonal of U
    diag_sum = 0.0
    for i in range(n):
        diag_sum += A[i * n + i]

    free(A)
    return diag_sum
