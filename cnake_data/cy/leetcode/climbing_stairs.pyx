# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sum of climbing stairs ways for 1..n using Fibonacci recurrence.

Keywords: leetcode, climbing stairs, fibonacci, dynamic programming, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(3000000,))
def climbing_stairs(int n):
    """Compute sum of ways(1) + ways(2) + ... + ways(n) mod 10^9+7."""
    cdef long long MOD = 1000000007
    cdef long long total = 0
    cdef long long a = 1, b = 1, temp
    cdef int i

    for i in range(n):
        total = (total + b) % MOD
        temp = b
        b = (a + b) % MOD
        a = temp

    return int(total)
