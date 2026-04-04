# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Vandermonde system solver for polynomial interpolation (Cython-optimized).

Keywords: numerical, vandermonde, polynomial, interpolation, linear algebra, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin, cos, fabs, M_PI
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(250,))
def vandermonde_solve(int n):
    """Solve a Vandermonde system for polynomial interpolation coefficients."""
    cdef int ncols = n + 1
    cdef double *M = <double *>malloc(n * ncols * sizeof(double))
    if not M:
        raise MemoryError()
    cdef double *nodes = <double *>malloc(n * sizeof(double))
    if not nodes:
        free(M)
        raise MemoryError()
    cdef double *c = <double *>malloc(n * sizeof(double))
    if not c:
        free(M)
        free(nodes)
        raise MemoryError()

    cdef int i, j, k, max_row
    cdef double val, max_val, pivot, factor, s, tmp, power
    cdef double c_sum, c_first, c_last

    # Generate Chebyshev nodes
    for i in range(n):
        if n > 1:
            nodes[i] = cos(i * M_PI / (n - 1))
        else:
            nodes[i] = 0.0

    # Build Vandermonde matrix V[i][j] = nodes[i]^j and RHS b[i]
    for i in range(n):
        power = 1.0
        for j in range(n):
            M[i * ncols + j] = power
            power *= nodes[i]
        M[i * ncols + n] = sin(nodes[i] + 0.5)

    # Gaussian elimination with partial pivoting
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
        c[i] = 0.0
    for i in range(n - 1, -1, -1):
        s = M[i * ncols + n]
        for j in range(i + 1, n):
            s -= M[i * ncols + j] * c[j]
        if fabs(M[i * ncols + i]) > 1e-15:
            c[i] = s / M[i * ncols + i]

    # Compute return values
    c_sum = 0.0
    for i in range(n):
        c_sum += c[i]
    c_first = c[0]
    c_last = c[n - 1]

    free(M)
    free(nodes)
    free(c)
    return (c_sum, c_first, c_last)
