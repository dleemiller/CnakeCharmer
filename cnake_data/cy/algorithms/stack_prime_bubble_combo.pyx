# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Combine prime generation and bubble-sort swap counting (Cython).

Adapted from The Stack v2 Cython candidate:
- blob_id: 81fffe9e4fe42d5fb755660a3ba982417a67b613
- filename: refactor_mod.pyx
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(7000, 1024))
def stack_prime_bubble_combo(int limit, int sort_n):
    cdef int *primes = <int *>malloc((limit + 1) * sizeof(int))
    cdef int prime_count = 0
    cdef int n, d
    cdef bint ok
    cdef int m, i, j, tmp
    cdef int *vals
    cdef long long swaps = 0
    cdef unsigned int checksum = 0

    if not primes:
        raise MemoryError()

    for n in range(2, limit + 1):
        ok = True
        d = 2
        while d * d <= n:
            if n % d == 0:
                ok = False
                break
            d += 1
        if ok:
            primes[prime_count] = n
            prime_count += 1

    m = sort_n if sort_n < prime_count else prime_count
    vals = <int *>malloc(m * sizeof(int))
    if not vals:
        free(primes)
        raise MemoryError()

    for i in range(m):
        vals[i] = (primes[i] * 37 + 11) % 10007

    for i in range(m):
        for j in range(0, m - i - 1):
            if vals[j] > vals[j + 1]:
                tmp = vals[j]
                vals[j] = vals[j + 1]
                vals[j + 1] = tmp
                swaps += 1

    for i in range(m):
        checksum = (checksum + <unsigned int>(vals[i] * (i + 3))) & 0xFFFFFFFF

    cdef int last_prime = 0
    if prime_count > 0:
        last_prime = primes[prime_count - 1]

    free(vals)
    free(primes)
    return (prime_count, last_prime, swaps, checksum)
