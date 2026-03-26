# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Sum of prime factors for all numbers from 2 to n (Cython-optimized).

Keywords: math, prime factorization, sieve, number theory, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def prime_factorization_sum(int n):
    """Compute total sum of prime factors using SPF sieve with C arrays."""
    cdef int i, j, x
    cdef long long total = 0
    cdef int max_factor_sum = 0
    cdef int factor_sum
    cdef int *spf = <int *>malloc((n + 1) * sizeof(int))

    if spf == NULL:
        raise MemoryError("Failed to allocate array")

    # Initialize spf[i] = i
    for i in range(n + 1):
        spf[i] = i

    # Build smallest prime factor sieve
    i = 2
    while i * i <= n:
        if spf[i] == i:  # i is prime
            j = i * i
            while j <= n:
                if spf[j] == j:
                    spf[j] = i
                j += i
        i += 1

    # Sum prime factors for each number
    for i in range(2, n + 1):
        x = i
        factor_sum = 0
        while x > 1:
            factor_sum += spf[x]
            x = x / spf[x]
        total += factor_sum
        if factor_sum > max_factor_sum:
            max_factor_sum = factor_sum

    cdef long long result_total = total
    cdef int result_max = max_factor_sum
    free(spf)
    return (result_total, result_max)
