# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Matrix multiplication (Cython-optimized with C arrays).

Keywords: matrix, multiply, linear algebra, numerical, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(150,))
def matrix_multiply(int n):
    """Multiply two n×n matrices using flat C arrays."""
    cdef double *A = <double *>malloc(n * n * sizeof(double))
    cdef double *B = <double *>malloc(n * n * sizeof(double))
    cdef double *C = <double *>malloc(n * n * sizeof(double))
    if not A or not B or not C:
        if A: free(A)
        if B: free(B)
        if C: free(C)
        raise MemoryError()

    cdef int i, j, k
    cdef double s

    # Initialize A[i][j] = i + j, B[i][j] = i - j
    for i in range(n):
        for j in range(n):
            A[i * n + j] = <double>(i + j)
            B[i * n + j] = <double>(i - j)

    memset(C, 0, n * n * sizeof(double))

    # Multiply
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i * n + k] * B[k * n + j]
            C[i * n + j] = s

    # Convert to Python list of lists
    cdef list result = []
    cdef list row
    for i in range(n):
        row = [C[i * n + j] for j in range(n)]
        result.append(row)

    free(A)
    free(B)
    free(C)
    return result
