# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute sum of square roots of 1..n using Newton's method (Cython-optimized).

Keywords: numerical, Newton's method, square root, iteration, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def newton_sqrt(int n):
    """Compute sum of square roots of 1..n using Newton's method (5 iterations each)."""
    cdef double total, x
    cdef int k, iteration

    total = 0.0
    for k in range(1, n + 1):
        x = k * 0.5
        for iteration in range(5):
            x = 0.5 * (x + k / x)
        total += x
    return total
