# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Prime testing with while-loop trial division (Cython-optimized).

Keywords: prime, while_loop, trial_division, primality, math, cython
"""

from cnake_charmer.benchmarks import cython_benchmark
from libc.math cimport sqrt


cdef inline bint _is_prime_while(int n):
    """Test primality using while loop."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    cdef int i = 3
    cdef int n_sqrt = <int>sqrt(<double>n) + 1
    while i < n_sqrt:
        if n % i == 0:
            return False
        i += 2
    return True


@cython_benchmark(syntax="cy", args=(200000,))
def count_primes_while(int n):
    """Count the number of primes up to n using while-loop trial division."""
    cdef int count = 0
    cdef int num
    for num in range(2, n + 1):
        if _is_prime_while(num):
            count += 1
    return count
