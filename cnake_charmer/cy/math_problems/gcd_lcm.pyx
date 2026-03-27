# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Compute GCD and LCM sums over all pairs in range(1, n) (Cython-optimized).

Keywords: gcd, lcm, math, pairs, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500,))
def gcd_lcm(int n):
    """Compute GCD and LCM sums using C-typed Euclidean algorithm."""
    cdef long long gcd_sum = 0
    cdef long long lcm_sum = 0
    cdef int i, j, a, b, g, temp

    for i in range(1, n):
        for j in range(i + 1, n):
            a = i
            b = j
            while b != 0:
                temp = b
                b = a % b
                a = temp
            g = a
            gcd_sum += g
            lcm_sum += <long long>i * <long long>j // g

    return (gcd_sum, lcm_sum)
