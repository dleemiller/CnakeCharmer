# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Gauss-Seidel iterative solver for a diagonally dominant system.

Keywords: numerical, linear algebra, iterative solver, gauss-seidel, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(300,))
def gauss_seidel(int n):
    """Solve an n x n diagonally dominant linear system using Gauss-Seidel."""
    cdef double *A = <double *>malloc(n * n * sizeof(double))
    cdef double *b = <double *>malloc(n * sizeof(double))
    cdef double *x = <double *>malloc(n * sizeof(double))
    if not A or not b or not x:
        if A: free(A)
        if b: free(b)
        if x: free(x)
        raise MemoryError()

    cdef int i, j, iteration
    cdef int max_iter = 200
    cdef double val, row_sum, sigma, solution_sum, x_first, x_last
    cdef double fn = <double>n

    # Build matrix
    for i in range(n):
        row_sum = 0.0
        for j in range(n):
            if i != j:
                val = ((i * 7 + j * 3) % 5) / fn
                A[i * n + j] = val
                row_sum += val
            else:
                A[i * n + j] = 0.0
        A[i * n + i] = 2.0 * fn
        b[i] = row_sum + 2.0 * fn

    # Initialize x to zero
    for i in range(n):
        x[i] = 0.0

    # Gauss-Seidel iteration
    for iteration in range(max_iter):
        for i in range(n):
            sigma = 0.0
            for j in range(n):
                if j != i:
                    sigma += A[i * n + j] * x[j]
            x[i] = (b[i] - sigma) / A[i * n + i]

    solution_sum = 0.0
    for i in range(n):
        solution_sum += x[i]

    x_first = x[0]
    x_last = x[n - 1]

    free(A)
    free(b)
    free(x)
    return (solution_sum, x_first, x_last)
