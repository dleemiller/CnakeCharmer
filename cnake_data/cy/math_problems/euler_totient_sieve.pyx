# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Compute Euler's totient function for all numbers 1..n using a sieve (Cython-optimized).

Keywords: math, euler, totient, sieve, number theory, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def euler_totient_sieve(int n):
    """Compute Euler's totient for 1..n using a sieve on C arrays."""
    cdef int i, j
    cdef int *phi

    if n < 1:
        return []

    phi = <int *>malloc((n + 1) * sizeof(int))
    if phi == NULL:
        raise MemoryError("Failed to allocate array")

    for i in range(n + 1):
        phi[i] = i

    for i in range(2, n + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, n + 1, i):
                phi[j] -= phi[j] / i

    cdef list result = [phi[i + 1] for i in range(n)]

    free(phi)
    return result
