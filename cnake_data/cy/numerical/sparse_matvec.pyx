# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sparse matrix-vector multiplication using CSR format (Cython-optimized).

Keywords: sparse matrix, CSR, matrix-vector multiply, numerical, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport llround
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def sparse_matvec(int n):
    """Build an n x n sparse matrix in CSR format and compute y = A * x.

    Args:
        n: Matrix dimension.

    Returns:
        Tuple of (int(y[0]*1e6), int(y[n//2]*1e6), int(sum(y)*1e3) % (10**9)).
    """
    cdef int nnz = n * 5
    cdef int *row_ptr = <int *>malloc((n + 1) * sizeof(int))
    cdef int *col_idx = <int *>malloc(nnz * sizeof(int))
    cdef double *values = <double *>malloc(nnz * sizeof(double))
    cdef double *x = <double *>malloc(n * sizeof(double))
    cdef double *y = <double *>malloc(n * sizeof(double))

    if not row_ptr or not col_idx or not values or not x or not y:
        free(row_ptr)
        free(col_idx)
        free(values)
        free(x)
        free(y)
        raise MemoryError()

    cdef int i, k, j, pos
    cdef double s, total
    cdef long long r0, r1, r2

    with nogil:
        # Build x
        for j in range(n):
            x[j] = j * 0.001

        # Build CSR
        for i in range(n):
            row_ptr[i] = i * 5
            for k in range(5):
                j = (i + k * 37) % n
                pos = i * 5 + k
                col_idx[pos] = j
                values[pos] = <double>((((<long long>i * j + 1) % 100) + 1))
        row_ptr[n] = nnz

        # Multiply y = A * x and zero y first
        for i in range(n):
            y[i] = 0.0

        for i in range(n):
            s = 0.0
            for pos in range(row_ptr[i], row_ptr[i + 1]):
                s = s + values[pos] * x[col_idx[pos]]
            y[i] = s

        total = 0.0
        for i in range(n):
            total = total + y[i]

        r0 = <long long>(y[0] * 1e6)
        r1 = <long long>(y[n // 2] * 1e6)
        r2 = llround(total * 1e3) % 1000000000

    free(row_ptr)
    free(col_idx)
    free(values)
    free(x)
    free(y)
    return (r0, r1, r2)
