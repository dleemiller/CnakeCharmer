# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Find abundant numbers up to n and compute their excess sums (Cython-optimized).

Keywords: abundant, numbers, divisors, sigma, number theory, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark

DEF MOD = 1000000007


@cython_benchmark(syntax="cy", args=(200000,))
def abundant_numbers(int n):
    """Count abundant numbers from 1 to n and sum their excesses."""
    cdef int i, j
    cdef int count = 0
    cdef long long excess_sum = 0
    cdef int last_abundant = 0

    cdef long long *sigma = <long long *>malloc((n + 1) * sizeof(long long))
    if sigma == NULL:
        raise MemoryError()

    for i in range(n + 1):
        sigma[i] = 0

    # Sieve for sum of proper divisors
    for i in range(1, n + 1):
        j = 2 * i
        while j <= n:
            sigma[j] += i
            j += i

    for i in range(2, n + 1):
        if sigma[i] > i:
            count += 1
            excess_sum = (excess_sum + sigma[i] - i) % MOD
            last_abundant = i

    free(sigma)
    return (count, int(excess_sum), last_abundant)
