# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Cython implementation of prime number generator.

Keywords: primes, algorithms, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(300,))
def primes(int nb_primes):
    """Generate a list of prime numbers using trial division.

    Args:
        nb_primes: Number of prime numbers to generate (capped at 1000).

    Returns:
        List of the first nb_primes prime numbers.
    """
    cdef int i
    cdef int p[1000]
    cdef int len_p = 0
    cdef int n = 2

    if nb_primes > 1000:
        nb_primes = 1000

    while len_p < nb_primes:
        for i in p[:len_p]:
            if n % i == 0:
                break
        else:
            p[len_p] = n
            len_p += 1
        n += 1

    return [p[i] for i in range(len_p)]
