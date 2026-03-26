# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count occurrences of pattern "AB" in the nth Fibonacci word using DP (Cython-optimized).

Keywords: dynamic programming, fibonacci, word, string, pattern, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000000,))
def fibonacci_word(int n):
    """Count 'AB' in nth Fibonacci word using pure typed arithmetic."""
    cdef long long MOD = 1000000007
    cdef long long count_prev2, count_prev1, count_cur
    cdef int first_prev2, last_prev2, first_prev1, last_prev1
    cdef int first_cur, last_cur, boundary
    cdef int i

    if n <= 0:
        return 0
    if n == 1:
        return 0
    if n == 2:
        return 1

    count_prev2 = 0
    first_prev2 = 0
    last_prev2 = 0

    count_prev1 = 1
    first_prev1 = 0
    last_prev1 = 1

    for i in range(3, n + 1):
        boundary = 1 if (last_prev1 == 0 and first_prev2 == 1) else 0
        count_cur = (count_prev1 + count_prev2 + boundary) % MOD
        first_cur = first_prev1
        last_cur = last_prev2

        count_prev2 = count_prev1
        first_prev2 = first_prev1
        last_prev2 = last_prev1

        count_prev1 = count_cur
        first_prev1 = first_cur
        last_prev1 = last_cur

    return count_prev1
