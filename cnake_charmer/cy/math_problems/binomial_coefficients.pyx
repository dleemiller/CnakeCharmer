# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute row n of binomial coefficients mod a prime using Pascal's triangle (Cython-optimized).

Keywords: binomial, pascal, triangle, combinatorics, coefficients, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark

DEF MOD = 1000000007


@cython_benchmark(syntax="cy", args=(3000,))
def binomial_coefficients(int n):
    """Compute row n of Pascal's triangle mod 10^9+7 and return summary statistics."""
    cdef int i, k
    cdef long long row_sum, middle, checksum
    cdef long long *prev
    cdef long long *curr
    cdef long long *tmp

    prev = <long long *>malloc((n + 1) * sizeof(long long))
    curr = <long long *>malloc((n + 1) * sizeof(long long))
    if prev == NULL or curr == NULL:
        if prev != NULL: free(prev)
        if curr != NULL: free(curr)
        raise MemoryError()

    for k in range(n + 1):
        prev[k] = 0
        curr[k] = 0
    prev[0] = 1

    for i in range(1, n + 1):
        curr[0] = 1
        for k in range(1, i + 1):
            curr[k] = (prev[k - 1] + prev[k]) % MOD
        for k in range(i + 1, n + 1):
            curr[k] = 0
        # Swap pointers
        tmp = prev
        prev = curr
        curr = tmp

    row_sum = 0
    for k in range(n + 1):
        row_sum = (row_sum + prev[k]) % MOD

    middle = prev[n // 2]

    checksum = 0
    for k in range(n + 1):
        checksum = (checksum + <long long>k * prev[k]) % MOD

    free(prev)
    free(curr)
    return (int(row_sum), int(middle), int(checksum))
