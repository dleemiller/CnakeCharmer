# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Fibonacci sequence generator (Cython-optimized).

Keywords: fibonacci, algorithms, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def fibonacci(int n):
    """Compute sum of first n Fibonacci numbers modulo 10^9+7."""
    cdef long long mod = 1000000007
    cdef long long a = 0, b = 1, total = 0, temp
    cdef int i

    for i in range(n):
        total = (total + b) % mod
        temp = (a + b) % mod
        a = b
        b = temp

    return int(total)
