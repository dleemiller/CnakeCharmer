# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute the Mobius function for 1..n using a sieve (Cython-optimized).

Keywords: mobius, number theory, sieve, multiplicative function, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def mobius_sieve(int n):
    """Compute Mobius function using sieve with C int arrays."""
    cdef int *mu = <int *>malloc((n + 1) * sizeof(int))
    cdef unsigned char *is_prime = <unsigned char *>malloc((n + 1) * sizeof(unsigned char))
    if not mu or not is_prime:
        free(mu)
        free(is_prime)
        raise MemoryError()

    cdef int i, j, total
    cdef long long i2

    memset(mu, 0, (n + 1) * sizeof(int))
    memset(is_prime, 1, (n + 1) * sizeof(unsigned char))
    mu[1] = 1

    for i in range(2, n + 1):
        if is_prime[i]:
            mu[i] = -1
            for j in range(2 * i, n + 1, i):
                is_prime[j] = 0
            for j in range(2 * i, n + 1, i):
                mu[j] = -mu[j]
            i2 = <long long>i * <long long>i
            if i2 <= n:
                j = <int>i2
                while j <= n:
                    mu[j] = 0
                    j += <int>i2

    total = 0
    for i in range(1, n + 1):
        total += mu[i]

    free(mu)
    free(is_prime)
    return total
