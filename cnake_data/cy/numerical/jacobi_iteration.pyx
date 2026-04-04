# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Solve tridiagonal system with Jacobi iteration (Cython-optimized).

Keywords: jacobi, iteration, linear algebra, tridiagonal, solver, numerical, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000,))
def jacobi_iteration(int n):
    """Solve Ax=b with Jacobi iteration on a tridiagonal matrix."""
    cdef int iterations = 1000
    cdef double *x = <double *>malloc(n * sizeof(double))
    cdef double *x_new = <double *>malloc(n * sizeof(double))
    cdef double *tmp_ptr

    if not x or not x_new:
        if x:
            free(x)
        if x_new:
            free(x_new)
        raise MemoryError()

    cdef int i, it
    cdef double sigma, total

    # Initialize x to 0
    for i in range(n):
        x[i] = 0.0

    for it in range(iterations):
        for i in range(n):
            sigma = 0.0
            if i > 0:
                sigma += x[i - 1]
            if i < n - 1:
                sigma += x[i + 1]
            x_new[i] = (1.0 - sigma) / (-2.0)
        # Swap pointers
        tmp_ptr = x
        x = x_new
        x_new = tmp_ptr

    total = 0.0
    for i in range(n):
        total += x[i]

    free(x)
    free(x_new)

    return total
