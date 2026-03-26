# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Matrix transpose of a flat n x n array (Cython-optimized).

Keywords: numerical, matrix, transpose, linear algebra, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(800,))
def matrix_transpose(int n):
    """Transpose an n x n matrix stored as a flat array."""
    cdef double *A = <double *>malloc(n * n * sizeof(double))
    if not A:
        raise MemoryError()

    cdef int i, j, idx_ij, idx_ji
    cdef double tmp, checksum, top_right, bottom_left

    # Build matrix
    for i in range(n):
        for j in range(n):
            A[i * n + j] = sin(i * 0.01 + j * 0.007)

    # Transpose in-place
    for i in range(n):
        for j in range(i + 1, n):
            idx_ij = i * n + j
            idx_ji = j * n + i
            tmp = A[idx_ij]
            A[idx_ij] = A[idx_ji]
            A[idx_ji] = tmp

    # Compute checksum
    checksum = 0.0
    for i in range(n * n):
        checksum += A[i]

    # Corner elements after transpose
    top_right = A[0 * n + (n - 1)]
    bottom_left = A[(n - 1) * n + 0]

    free(A)
    return (checksum, top_right, bottom_left)
