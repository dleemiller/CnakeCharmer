# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute sum of digit sums of all numbers from 1 to n (Cython-optimized).

Keywords: math, digits, sum, enumeration, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2000000,))
def digit_sum(int n):
    """Compute total digit sum using pure typed arithmetic."""
    cdef long long total = 0
    cdef int i, num

    for i in range(1, n + 1):
        num = i
        while num > 0:
            total += num % 10
            num = num / 10

    return total
