# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Matrix multiplication returning trace, corner sum, and mid element.

Keywords: matrix, multiply, linear algebra, numerical, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(150,))
def matrix_multiply(int n):
    """Multiply two n x n matrices using flat C arrays and return summary."""
    cdef double *A = <double *>malloc(n * n * sizeof(double))
    cdef double *B = <double *>malloc(n * n * sizeof(double))
    cdef double *C = <double *>malloc(n * n * sizeof(double))
    if not A or not B or not C:
        if A: free(A)
        if B: free(B)
        if C: free(C)
        raise MemoryError()

    cdef int i, j, k, mid_idx
    cdef double s, trace, corner_sum, mid_element

    for i in range(n):
        for j in range(n):
            A[i * n + j] = <double>(i + j)
            B[i * n + j] = <double>(i - j)

    memset(C, 0, n * n * sizeof(double))

    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i * n + k] * B[k * n + j]
            C[i * n + j] = s

    trace = 0.0
    for i in range(n):
        trace += C[i * n + i]

    corner_sum = C[0] + C[n - 1] + C[(n - 1) * n] + C[(n - 1) * n + n - 1]
    mid_idx = n // 2
    mid_element = C[mid_idx * n + mid_idx]

    free(A)
    free(B)
    free(C)
    return (trace, corner_sum, mid_element)
