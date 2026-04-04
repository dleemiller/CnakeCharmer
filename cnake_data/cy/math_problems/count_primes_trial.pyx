# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Prime counting via trial division (Cython)."""

from cnake_data.benchmarks import cython_benchmark


cdef inline bint _is_prime(int x) noexcept:
    cdef int d
    if x < 2:
        return False
    if x == 2:
        return True
    if x % 2 == 0:
        return False
    d = 3
    while d * d <= x:
        if x % d == 0:
            return False
        d += 2
    return True


@cython_benchmark(syntax="cy", args=(6000, True))
def count_primes_trial(int limit, bint return_last):
    cdef int count = 0
    cdef int last = 0
    cdef int x
    for x in range(2, limit + 1):
        if _is_prime(x):
            count += 1
            last = x
    if not return_last:
        last = 0
    return (count, last)

