# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Solve a tridiagonal system using the Thomas algorithm (Cython-optimized).

Keywords: tridiagonal, linear system, Thomas algorithm, numerical, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def tridiagonal_solve(int n):
    """Solve tridiagonal system Ax=b using Thomas algorithm with C arrays."""
    cdef double *cp = <double *>malloc(n * sizeof(double))
    cdef double *dp = <double *>malloc(n * sizeof(double))
    if not cp or not dp:
        if cp: free(cp)
        if dp: free(dp)
        raise MemoryError()

    cdef int i
    cdef double rhs_i, denom, result, total

    # First row
    cp[0] = 1.0 / 4.0
    dp[0] = sin(0.0) / 4.0

    for i in range(1, n):
        rhs_i = sin(i * 0.1)
        denom = 4.0 - 1.0 * cp[i - 1]
        if i < n - 1:
            cp[i] = 1.0 / denom
        else:
            cp[i] = 0.0
        dp[i] = (rhs_i - 1.0 * dp[i - 1]) / denom

    # Back substitution
    result = dp[n - 1]
    total = result
    for i in range(n - 2, -1, -1):
        result = dp[i] - cp[i] * result
        total += result

    free(cp)
    free(dp)
    return total
