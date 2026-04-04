# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sum primes via trial division (Cython).

Sourced from SFT DuckDB blob: 14834b7d32cd6d788819f5842484cf3bd5313aa9
Keywords: prime, trial division, prime summation, algorithms, cython
"""

from cnake_charmer.benchmarks import cython_benchmark


cdef inline bint _is_prime_trial(int n):
    cdef int d
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


@cython_benchmark(syntax="cy", args=(150000, 2, 1))
def prime_sum_trial_division(int limit, int start, int step):
    cdef int x
    cdef long long total = 0
    cdef int count = 0
    cdef int last_prime = -1

    for x in range(start, limit, step):
        if _is_prime_trial(x):
            total += x
            count += 1
            last_prime = x

    return (total, count, last_prime)

