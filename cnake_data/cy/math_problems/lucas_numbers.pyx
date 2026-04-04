# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute sum of first n Lucas numbers mod 10^9+7 (Cython-optimized).

Keywords: lucas, fibonacci, sequence, number theory, modular arithmetic, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(3000000,))
def lucas_numbers(int n):
    """Compute sum of first n Lucas numbers mod 10^9+7 using typed arithmetic."""
    cdef long long MOD = 1000000007
    cdef long long prev2, prev1, curr, total
    cdef int i

    if n == 0:
        return 0
    if n == 1:
        return 2

    prev2 = 2  # L(0)
    prev1 = 1  # L(1)
    total = prev2 + prev1

    for i in range(2, n):
        curr = (prev1 + prev2) % MOD
        total = (total + curr) % MOD
        prev2 = prev1
        prev1 = curr

    return total % MOD
