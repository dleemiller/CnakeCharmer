# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Parallel prime counting using prange with += reduction.

Keywords: math, primes, counting, prange, parallel, cython, benchmark
"""

from cython.parallel import prange
from cnake_data.benchmarks import cython_benchmark


cdef inline int _is_prime(int num) noexcept nogil:
    """Test if num is prime. Returns 1 if prime, 0 otherwise."""
    cdef int d
    if num < 2:
        return 0
    if num == 2:
        return 1
    if num % 2 == 0:
        return 0
    d = 3
    while d * d <= num:
        if num % d == 0:
            return 0
        d += 2
    return 1


@cython_benchmark(syntax="cy", args=(100000,))
def prange_count_primes(int n):
    """Count primes in [2, n] with prange += reduction."""
    cdef int num
    cdef int count = 0

    if n < 2:
        return 0

    for num in prange(2, n + 1, nogil=True):
        count += _is_prime(num)

    return count
