# cython: boundscheck=False, wraparound=False, language_level=3
"""Fibonacci sequence generator (Cython-optimized).

Keywords: fibonacci, algorithms, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1e18,))
def fib(long long n):
    """Compute Fibonacci numbers less than n.

    Args:
        n: Upper limit for the sequence (exclusive).

    Returns:
        List of Fibonacci numbers less than n.
    """
    cdef long long a = 0
    cdef long long b = 1
    cdef list result = []

    while b < n:
        result.append(b)
        a, b = b, a + b

    return result
