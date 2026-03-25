# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute the nth row of Pascal's triangle (Cython-optimized).

Keywords: pascal, triangle, binomial, combinatorics, math, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2000,))
def pascal_triangle_row(int n):
    """Compute the nth row of Pascal's triangle.

    Uses Python ints for arbitrary precision (values exceed long long
    for n > ~60), but typed loop variables for speed.
    """
    cdef int i, j
    cdef int size = n + 1
    cdef list row = [1] * size

    for i in range(2, n + 1):
        prev = 1
        for j in range(1, i):
            temp = row[j]
            row[j] = prev + row[j]
            prev = temp

    return row
