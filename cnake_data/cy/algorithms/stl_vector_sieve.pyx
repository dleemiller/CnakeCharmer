# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sieve of Eratosthenes collecting primes into a C++ vector.

Keywords: sieve, primes, Eratosthenes, libcpp, vector, benchmark
"""

from libcpp.vector cimport vector
from cnake_data.benchmarks import cython_benchmark

DEF MOD = 1000000007


@cython_benchmark(syntax="cy", args=(500000,))
def stl_vector_sieve(int n):
    """Sieve of Eratosthenes using a C++ vector[int] for prime collection.

    Args:
        n: Upper bound (inclusive) for the sieve.

    Returns:
        Tuple of (num_primes, weighted_sum_mod) where weighted_sum_mod is
        sum(prime[i] * (i+1) for i in range(len(primes))) % (10**9 + 7).
    """
    cdef int i, j
    cdef vector[bint] sieve
    sieve.resize(n + 1, 1)
    if n >= 0:
        sieve[0] = 0
    if n >= 1:
        sieve[1] = 0

    i = 2
    while i * i <= n:
        if sieve[i]:
            j = i * i
            while j <= n:
                sieve[j] = 0
                j += i
        i += 1

    cdef vector[int] primes
    primes.reserve(n // 10 + 10)
    for i in range(2, n + 1):
        if sieve[i]:
            primes.push_back(i)

    cdef int num_primes = primes.size()
    cdef long long weighted_sum = 0
    for i in range(num_primes):
        weighted_sum = (weighted_sum + <long long>primes[i] * (i + 1)) % MOD

    return (num_primes, <int>weighted_sum)
