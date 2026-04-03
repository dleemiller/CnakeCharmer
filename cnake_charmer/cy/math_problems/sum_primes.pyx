# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sum of all primes less than n using trial division.

Keywords: primes, sum, trial division, math, cython, benchmark
"""

from libc.math cimport ceil, sqrt
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(50000,))
def sum_primes(int n):
    """Sum all primes less than n using trial division."""
    cdef unsigned long total = 0
    cdef int count = 0
    cdef int x, i, max_i
    cdef int is_prime_flag

    with nogil:
        for x in range(2, n):
            if x == 2:
                is_prime_flag = 1
            elif x % 2 == 0:
                is_prime_flag = 0
            else:
                is_prime_flag = 1
                max_i = <int>ceil(sqrt(<double>x))
                i = 3
                while i <= max_i:
                    if x % i == 0:
                        is_prime_flag = 0
                        break
                    i += 2

            if is_prime_flag:
                total += x
                count += 1

    return (<long long>total, count)
