# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Element-wise addition of two n*n matrices using 2D typed memoryviews, returning the sum of all elements.

Keywords: numerical, matrix, addition, element-wise, typed memoryview, cython, benchmark
"""

from cython.view cimport array as cvarray
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(300,))
def memview_mat_add(int n):
    """Add two n*n matrices element-wise using 2D memoryviews, return total sum."""
    cdef int i, j
    cdef double total

    arr_a = cvarray(shape=(n, n), itemsize=sizeof(double), format="d")
    cdef double[:, :] A = arr_a

    arr_b = cvarray(shape=(n, n), itemsize=sizeof(double), format="d")
    cdef double[:, :] B = arr_b

    for i in range(n):
        for j in range(n):
            A[i, j] = ((i * 53 + j * 37 + 7) % 500) / 5.0
            B[i, j] = ((i * 41 + j * 67 + 13) % 500) / 5.0

    total = 0.0
    for i in range(n):
        for j in range(n):
            total += A[i, j] + B[i, j]

    return total
