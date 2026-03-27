# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sieve of Eratosthenes returning count, largest prime, and sum mod 10^9+7.

Keywords: sieve, primes, number theory, math, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2000000,))
def prime_sieve(int n):
    """Sieve primes up to n and return summary statistics."""
    if n < 2:
        return (0, 0, 0)

    cdef char *is_prime = <char *>malloc((n + 1) * sizeof(char))
    if not is_prime:
        raise MemoryError()

    cdef int i, j
    cdef int count = 0
    cdef int largest = 0
    cdef long long total = 0
    cdef long long MOD = 1000000007

    for i in range(n + 1):
        is_prime[i] = 1
    is_prime[0] = 0
    is_prime[1] = 0

    i = 2
    while i * i <= n:
        if is_prime[i]:
            j = i * i
            while j <= n:
                is_prime[j] = 0
                j += i
        i += 1

    for i in range(2, n + 1):
        if is_prime[i]:
            count += 1
            largest = i
            total = (total + i) % MOD

    free(is_prime)
    return (count, largest, <int>(total))
