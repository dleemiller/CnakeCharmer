# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute sum of Fibonacci numbers F(1)+F(2)+...+F(n) (Cython-optimized).

Keywords: fibonacci, modular arithmetic, number theory, accumulation, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark

DEF MOD = 1000000007


@cython_benchmark(syntax="cy", args=(3000000,))
def fibonacci_matrix(int n):
    """Compute sum F(1)+F(2)+...+F(n) mod 10^9+7."""
    cdef long long a, b, tmp, total
    cdef int i

    a = 0
    b = 1
    total = 0
    for i in range(n):
        tmp = (a + b) % MOD
        a = b
        b = tmp
        total = (total + a) % MOD

    return int(total)
