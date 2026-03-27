# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Transpose an n*n matrix and compute the trace of A + A^T using 2D typed memoryviews.

Keywords: numerical, matrix, transpose, trace, typed memoryview, cython, benchmark
"""

from cython.view cimport array as cvarray
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(300,))
def memview_mat_transpose(int n):
    """Transpose n*n matrix using 2D memoryviews, return trace of A + A^T."""
    cdef int i, j
    cdef double trace

    # Allocate and fill matrix A using cvarray for 2D memoryview
    arr_a = cvarray(shape=(n, n), itemsize=sizeof(double), format="d")
    cdef double[:, :] A = arr_a

    arr_at = cvarray(shape=(n, n), itemsize=sizeof(double), format="d")
    cdef double[:, :] AT = arr_at

    for i in range(n):
        for j in range(n):
            A[i, j] = ((i * 97 + j * 31 + 17) % 1000) / 10.0

    # Transpose
    for i in range(n):
        for j in range(n):
            AT[i, j] = A[j, i]

    # Trace of A + AT
    trace = 0.0
    for i in range(n):
        trace += A[i, i] + AT[i, i]

    return trace
