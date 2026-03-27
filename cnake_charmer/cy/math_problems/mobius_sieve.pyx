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

    with nogil:
        i = 2
        while i <= n:
            if is_prime[i]:
                mu[i] = -1
                # Fused single pass: mark composites and flip mu together
                j = 2 * i
                while j <= n:
                    is_prime[j] = 0
                    mu[j] = -mu[j]
                    j += i
                # Zero out multiples of i^2
                i2 = <long long>i * <long long>i
                if i2 <= n:
                    j = <int>i2
                    while j <= n:
                        mu[j] = 0
                        j += <int>i2
            i += 1

        total = 0
        j = 1
        while j <= n:
            total += mu[j]
            j += 1

    free(mu)
    free(is_prime)
    return total
