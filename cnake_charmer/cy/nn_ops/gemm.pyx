# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""General matrix multiply (basic Cython, no SIMD).

Keywords: matrix multiply, gemm, linear algebra, neural network, cython
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(300,))
def gemm(int n):
    """Compute C = A * B for n×n matrices, return trace(C). Basic triple loop."""
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

    for i in range(n):
        for j in range(n):
            A[i * n + j] = ((i + j) % 100) / 10.0
            B[i * n + j] = ((i - j + n) % 100) / 10.0

    memset(C, 0, n * n * sizeof(double))

    for i in range(n):
        for k in range(n):
            s = A[i * n + k]
            for j in range(n):
                C[i * n + j] += s * B[k * n + j]

    cdef double trace = 0.0
    for i in range(n):
        trace += C[i * n + i]

    free(A)
    free(B)
    free(C)
    return trace
