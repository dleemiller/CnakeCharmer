# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sum all primes below n using trial division (Cython-optimized)."""

from libc.math cimport sqrt
from cnake_charmer.benchmarks import cython_benchmark


cdef inline bint _is_prime(int x) noexcept nogil:
    """Trial division primality test."""
    cdef int d
    cdef int limit
    if x < 2:
        return 0
    if x == 2:
        return 1
    if x % 2 == 0:
        return 0
    limit = <int>sqrt(<double>x) + 1
    d = 3
    while d < limit:
        if x % d == 0:
            return 0
        d += 2
    return 1


@cython_benchmark(syntax="cy", args=(100000,))
def prime_sum(int n):
    """Sum and count all prime numbers less than n using trial division.

    Args:
        n: Upper bound (exclusive).

    Returns:
        (prime_sum, prime_count) — sum and count of primes < n.
    """
    cdef int x
    cdef long total = 0
    cdef int count = 0

    with nogil:
        for x in range(2, n):
            if _is_prime(x):
                total += x
                count += 1

    return (int(total), int(count))
