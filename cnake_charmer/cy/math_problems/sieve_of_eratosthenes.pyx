# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Sieve of Eratosthenes to find all primes up to n (Cython-optimized).

Keywords: sieve, eratosthenes, primes, math, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark
import cython


@cython_benchmark(syntax="cy", args=(1000000,))
def sieve_of_eratosthenes(int n):
    """Find all primes up to n using a C-typed Sieve of Eratosthenes."""
    cdef int i, j
    cdef list is_prime
    cdef list result

    if n < 2:
        return []

    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False

    i = 2
    while i * i <= n:
        if is_prime[i]:
            j = i * i
            while j <= n:
                is_prime[j] = False
                j += i
        i += 1

    result = []
    for i in range(2, n + 1):
        if is_prime[i]:
            result.append(i)

    return result
