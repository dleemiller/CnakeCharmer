# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sparse matrix-vector multiply in CSR format (Cython-optimized).

Keywords: sparse matrix, CSR, matrix-vector multiply, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def sparse_matrix_vector(int n):
    """Sparse matrix-vector multiply in CSR format using C arrays."""
    cdef int nnz = 3 * n
    cdef int *row_ptr = <int *>malloc((n + 1) * sizeof(int))
    cdef int *col_idx = <int *>malloc(nnz * sizeof(int))
    cdef double *vals = <double *>malloc(nnz * sizeof(double))
    cdef double *vec = <double *>malloc(n * sizeof(double))

    if not row_ptr or not col_idx or not vals or not vec:
        if row_ptr: free(row_ptr)
        if col_idx: free(col_idx)
        if vals: free(vals)
        if vec: free(vec)
        raise MemoryError()

    cdef int i, j, base
    cdef double row_sum, total
    cdef double *result = <double *>malloc(n * sizeof(double))
    cdef double result_at_0, result_at_half

    if not result:
        free(row_ptr)
        free(col_idx)
        free(vals)
        free(vec)
        raise MemoryError()

    # Build CSR
    row_ptr[0] = 0
    for i in range(n):
        row_ptr[i + 1] = row_ptr[i] + 3
        base = i * 3
        col_idx[base] = (i * 3 + 1) % n
        col_idx[base + 1] = (i * 7 + 2) % n
        col_idx[base + 2] = (i * 11 + 3) % n
        vals[base] = 1.0
        vals[base + 1] = 2.0
        vals[base + 2] = 3.0

    # Vector
    for i in range(n):
        vec[i] = i * 0.1

    # SpMV
    total = 0.0
    for i in range(n):
        row_sum = 0.0
        for j in range(row_ptr[i], row_ptr[i + 1]):
            row_sum += vals[j] * vec[col_idx[j]]
        result[i] = row_sum
        total += row_sum

    result_at_0 = result[0]
    result_at_half = result[n / 2]

    free(result)
    free(row_ptr)
    free(col_idx)
    free(vals)
    free(vec)
    return (total, result_at_0, result_at_half)
