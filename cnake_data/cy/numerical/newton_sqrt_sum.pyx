# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Newton's method square root sum — Cython implementation."""

from libc.math cimport fabs

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000,))
def newton_sqrt_sum(int n):
    """Sum of Newton's-method square roots for integers 1..n.

    Args:
        n: Upper bound (inclusive).

    Returns:
        Tuple of (total_sum, midpoint_sqrt, last_sqrt).
    """
    cdef int v, k, mid
    cdef double total = 0.0
    cdef double mid_val = 0.0, last_val = 0.0
    cdef double x, xprev, fv, diff

    mid = n // 2
    for v in range(1, n + 1):
        x = 1.0
        fv = <double>v
        for k in range(100):
            xprev = x
            x = (x + fv / x) * 0.5
            diff = fabs(x - xprev)
            if diff < 1e-14:
                break
        total += x
        if v == mid:
            mid_val = x
        last_val = x
    return (total, mid_val, last_val)
