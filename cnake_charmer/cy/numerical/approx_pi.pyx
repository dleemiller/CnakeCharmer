# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Monte Carlo pi approximation using a fixed-seed LCG random number generator (Cython-optimized).

Keywords: monte carlo, pi, numerical, random, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark
import cython


@cython_benchmark(syntax="cy", args=(100000,))
def approx_pi(int n):
    """Approximate pi using Monte Carlo method with C-typed LCG PRNG."""
    cdef int i, inside = 0
    cdef long long seed = 42
    cdef long long a = 1103515245
    cdef long long c = 12345
    cdef long long m = 2147483648
    cdef double x, y

    for i in range(n):
        seed = (a * seed + c) % m
        x = <double>seed / <double>m
        seed = (a * seed + c) % m
        y = <double>seed / <double>m
        if x * x + y * y <= 1.0:
            inside += 1

    return 4.0 * inside / n
