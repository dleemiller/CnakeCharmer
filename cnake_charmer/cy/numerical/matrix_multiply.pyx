# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Matrix multiplication (Cython-optimized).

Keywords: matrix, multiply, linear algebra, numerical, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark
import cython


@cython_benchmark(syntax="cy", args=(50,))
def matrix_multiply(int n):
    """Multiply two n×n matrices using typed memoryviews."""
    cdef int i, j, k
    cdef double s

    cdef list A = [[float(i + j) for j in range(n)] for i in range(n)]
    cdef list B = [[float(i - j) for j in range(n)] for i in range(n)]
    cdef list C = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += <double>A[i][k] * <double>B[k][j]
            C[i][j] = s

    return C
