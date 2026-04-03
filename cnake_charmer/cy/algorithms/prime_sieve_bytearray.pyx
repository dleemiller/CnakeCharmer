# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sieve of Eratosthenes using a bytearray for compact prime flagging.

Marks composite numbers in a half-sieve (odd numbers only) using a bytearray,
then counts the surviving primes. This approach avoids storing all primes
in a list and uses minimal memory per candidate.

Keywords: algorithms, prime sieve, Eratosthenes, bytearray, number theory, cython, benchmark
"""

from libc.math cimport sqrt

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def prime_sieve_bytearray(int limit):
    """Count primes up to limit using a bytearray sieve."""
    if limit < 2:
        return (0, 0)

    cdef int size = (limit + 1) / 2
    cdef bytearray bits_arr = bytearray(b'\x01') * size
    cdef char *bits = <char *>bits_arr

    cdef int factor = 1
    cdef double q = sqrt(<double>limit) / 2.0
    cdef int index, start, step, idx
    cdef int prime_count, i
    cdef long long prime_sum

    while factor <= q:
        # Find next prime
        for index in range(factor, size):
            if bits[index]:
                factor = index
                break

        # Mark multiples starting at factor^2
        start = 2 * factor * (factor + 1)
        step = factor * 2 + 1

        idx = start
        while idx < size:
            bits[idx] = 0
            idx += step

        factor += 1

    # Count primes and compute sum
    prime_count = 1  # count 2
    prime_sum = 2
    for i in range(1, size):
        if bits[i]:
            prime_count += 1
            prime_sum += 2 * i + 1

    return (prime_count, prime_sum)
