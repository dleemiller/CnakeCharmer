# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count primes in first n integers using trial division
(Cython-optimized with cpdef standalone function).

Keywords: prime, counting, math, cpdef, standalone function, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


cpdef bint is_prime(long long n):
    """Check if n is prime using trial division."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    cdef long long i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


@cython_benchmark(syntax="cy", args=(50000,))
def cpdef_is_prime_count(int n):
    """Count how many integers in [0, n) are prime."""
    cdef int count = 0
    cdef int i
    for i in range(n):
        if is_prime(<long long>i):
            count += 1
    return count
