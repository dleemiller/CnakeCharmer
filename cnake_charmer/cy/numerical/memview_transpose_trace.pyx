# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Trace of A * A^T element-wise via memoryview .T attribute.

Fills matrix, gets transposed view via .T, computes sum of
element-wise product A[i,j] * A^T[i,j].

Keywords: numerical, matrix, transpose, trace, memoryview, cython, benchmark
"""

from cython.view cimport array as cvarray
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200,))
def memview_transpose_trace(int n):
    """Compute element-wise product sum of A and A^T."""
    cdef int i, j, idx
    cdef unsigned int h
    cdef double trace

    arr = cvarray(
        shape=(n, n),
        itemsize=sizeof(double),
        format="d",
    )
    cdef double[:, :] mat = arr

    for i in range(n):
        for j in range(n):
            idx = i * n + j
            h = (
                (<unsigned int>idx
                 * <unsigned int>2654435761)
                ^ (<unsigned int>idx
                   * <unsigned int>2246822519)
            )
            mat[i, j] = (
                <double>(h & 0xFFFF) / 65535.0
            )

    cdef double[:, :] mat_t = mat.T

    trace = 0.0
    for i in range(n):
        for j in range(n):
            trace += mat[i, j] * mat_t[i, j]
    return trace
