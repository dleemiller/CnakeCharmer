# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute trace of matrix power (Cython with typed memoryviews).

Keywords: matrix, power, trace, typed memoryview, linear algebra, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


cdef void mat_mul(double[:] X, double[:] Y, double[:] R, int size):
    """Multiply two matrices stored as 1D memoryviews in row-major order."""
    cdef int i, j, k
    cdef double x_ik

    for i in range(size * size):
        R[i] = 0.0

    for i in range(size):
        for k in range(size):
            x_ik = X[i * size + k]
            if x_ik == 0.0:
                continue
            for j in range(size):
                R[i * size + j] += x_ik * Y[k * size + j]


@cython_benchmark(syntax="cy", args=(150,))
def matrix_power_trace(int n):
    """Compute trace(A^4) using typed memoryviews for matrix storage."""
    cdef int nn = n * n
    cdef double *a_ptr = <double *>malloc(nn * sizeof(double))
    cdef double *a2_ptr = <double *>malloc(nn * sizeof(double))
    cdef double *a4_ptr = <double *>malloc(nn * sizeof(double))
    if not a_ptr or not a2_ptr or not a4_ptr:
        if a_ptr: free(a_ptr)
        if a2_ptr: free(a2_ptr)
        if a4_ptr: free(a4_ptr)
        raise MemoryError()

    # Create typed memoryviews from raw pointers
    cdef double[:] A = <double[:nn]>a_ptr
    cdef double[:] A2 = <double[:nn]>a2_ptr
    cdef double[:] A4 = <double[:nn]>a4_ptr

    cdef int i, j
    cdef unsigned long long h
    cdef double trace

    # Generate matrix
    for i in range(n):
        for j in range(n):
            h = ((<unsigned long long>i * 2654435761 + <unsigned long long>j * 1103515245) >> 12) & 0xFFF
            A[i * n + j] = (<int>(h % 201) - 100) / 100.0

    # Compute A^4 = (A*A) * (A*A)
    mat_mul(A, A, A2, n)
    mat_mul(A2, A2, A4, n)

    # Trace
    trace = 0.0
    for i in range(n):
        trace += A4[i * n + i]

    free(a_ptr)
    free(a2_ptr)
    free(a4_ptr)
    return trace
