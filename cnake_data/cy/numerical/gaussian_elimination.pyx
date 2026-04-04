# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Gaussian elimination with partial pivoting (Cython-optimized).

Keywords: numerical, gaussian elimination, linear algebra, solver, pivoting, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin, cos, fabs
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(250,))
def gaussian_elimination(int n):
    """Solve Ax = b via Gaussian elimination with partial pivoting."""
    cdef int ncols = n + 1
    cdef double *M = <double *>malloc(n * ncols * sizeof(double))
    if not M:
        raise MemoryError()
    cdef double *x = <double *>malloc(n * sizeof(double))
    if not x:
        free(M)
        raise MemoryError()

    cdef int i, j, k, max_row
    cdef double val, max_val, pivot, factor, s, tmp, x_sum, x_first, x_last

    # Build augmented matrix [A | b]
    for i in range(n):
        for j in range(n):
            val = sin(i + j + 1)
            if i == j:
                val = val + 2.0
            M[i * ncols + j] = val
        M[i * ncols + n] = cos(i * 0.3)

    # Forward elimination with partial pivoting
    for k in range(n):
        # Find pivot
        max_val = fabs(M[k * ncols + k])
        max_row = k
        for i in range(k + 1, n):
            val = fabs(M[i * ncols + k])
            if val > max_val:
                max_val = val
                max_row = i

        # Swap rows
        if max_row != k:
            for j in range(ncols):
                tmp = M[k * ncols + j]
                M[k * ncols + j] = M[max_row * ncols + j]
                M[max_row * ncols + j] = tmp

        # Eliminate below
        pivot = M[k * ncols + k]
        if fabs(pivot) < 1e-15:
            continue
        for i in range(k + 1, n):
            factor = M[i * ncols + k] / pivot
            for j in range(k, ncols):
                M[i * ncols + j] -= factor * M[k * ncols + j]

    # Back substitution
    for i in range(n):
        x[i] = 0.0
    for i in range(n - 1, -1, -1):
        s = M[i * ncols + n]
        for j in range(i + 1, n):
            s -= M[i * ncols + j] * x[j]
        if fabs(M[i * ncols + i]) > 1e-15:
            x[i] = s / M[i * ncols + i]

    # Compute return values
    x_sum = 0.0
    for i in range(n):
        x_sum += x[i]
    x_first = x[0]
    x_last = x[n - 1]

    free(M)
    free(x)
    return (x_sum, x_first, x_last)
