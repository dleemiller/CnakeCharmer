# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute the nth row of Pascal's triangle (Cython-optimized).

Keywords: pascal, triangle, binomial, combinatorics, math, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2000,))
def pascal_triangle_row(int n):
    """Compute the nth row of Pascal's triangle.

    Uses the O(n) binomial coefficient recurrence:
        C(n, k) = C(n, k-1) * (n - k + 1) // k
    This computes each element in one pass with a single multiply and
    divide per step instead of the O(n^2) inner loop.

    Uses Python ints for arbitrary precision (values exceed long long
    for n > ~60), but typed loop variables for speed.
    """
    cdef int k
    cdef list row = [None] * (n + 1)

    row[0] = 1
    val = 1
    for k in range(1, n + 1):
        val = val * (n - k + 1) // k
        row[k] = val

    return row
