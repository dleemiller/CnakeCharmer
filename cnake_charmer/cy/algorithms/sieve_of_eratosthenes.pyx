# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sieve of Eratosthenes to find all primes up to n (Cython-optimized).

Keywords: algorithms, sieve, primes, number theory, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from libc.math cimport sqrt
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000000,))
def sieve_of_eratosthenes(int n):
    """Find all primes up to n using the Sieve of Eratosthenes.

    Returns:
        Tuple of (prime_count, prime_sum % 10**9).
    """
    cdef unsigned char *sieve = <unsigned char *>malloc((n + 1) * sizeof(unsigned char))
    if not sieve:
        raise MemoryError()

    cdef int i, j, sqrtn
    cdef long long prime_count, prime_sum

    with nogil:
        memset(sieve, 1, (n + 1) * sizeof(unsigned char))
        sieve[0] = 0
        if n >= 1:
            sieve[1] = 0

        sqrtn = <int>sqrt(<double>n)
        i = 2
        while i <= sqrtn:
            if sieve[i]:
                j = i * i
                while j <= n:
                    sieve[j] = 0
                    j += i
            i += 1

        prime_count = 0
        prime_sum = 0
        for i in range(2, n + 1):
            if sieve[i]:
                prime_count += 1
                prime_sum += i

    free(sieve)
    return (prime_count, prime_sum % (10 ** 9))
