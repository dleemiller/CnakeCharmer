# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute sum of Euler's totient phi(1)+...+phi(n) using a sieve (Cython-optimized).

Keywords: math, euler, totient, sieve, sum, number theory, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def totient_sum(int n):
    """Compute totient sum using C array sieve."""
    cdef int i, j
    cdef int *phi
    cdef long long total

    if n < 1:
        return 0

    phi = <int *>malloc((n + 1) * sizeof(int))
    if not phi:
        raise MemoryError()

    for i in range(n + 1):
        phi[i] = i

    for i in range(2, n + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, n + 1, i):
                phi[j] -= phi[j] / i

    total = 0
    for i in range(1, n + 1):
        total += phi[i]

    free(phi)
    return total
