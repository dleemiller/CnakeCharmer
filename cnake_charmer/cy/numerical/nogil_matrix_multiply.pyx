# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Matrix multiply with GIL release, returning trace.

Keywords: matrix, multiply, nogil, numerical, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


cdef double _matmul_trace(
    double *A, double *B, int n
) noexcept nogil:
    """Compute trace of A * B without GIL."""
    cdef double trace = 0.0
    cdef double s
    cdef int i, k
    for i in range(n):
        s = 0.0
        for k in range(n):
            s += A[i * n + k] * B[k * n + i]
        trace += s
    return trace


@cython_benchmark(syntax="cy", args=(200,))
def nogil_matrix_multiply(int n):
    """Multiply two n x n matrices, return trace."""
    cdef int size = n * n
    cdef double *A = <double *>malloc(
        size * sizeof(double)
    )
    cdef double *B = <double *>malloc(
        size * sizeof(double)
    )
    if not A or not B:
        if A:
            free(A)
        if B:
            free(B)
        raise MemoryError()

    cdef int i, j
    for i in range(n):
        for j in range(n):
            A[i * n + j] = (
                ((i * 3 + j * 7 + 1) % 100) / 10.0
            )
            B[i * n + j] = (
                ((i * 11 + j * 5 + 3) % 100) / 10.0
            )

    cdef double trace
    with nogil:
        trace = _matmul_trace(A, B, n)

    free(A)
    free(B)
    return trace
