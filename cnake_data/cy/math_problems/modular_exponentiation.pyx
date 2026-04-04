# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Sum of modular exponentiations using binary exponentiation (Cython-optimized).

Keywords: math, modular exponentiation, binary exponentiation, number theory, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def modular_exponentiation(int n):
    """Compute sum of modular exponentiations with typed variables."""
    cdef long long MOD = 1000000007
    cdef long long total = 0
    cdef long long result, b
    cdef int i, base, exp, e

    for i in range(n):
        base = (i * 7 + 3) % 1000
        exp = (i * 13 + 7) % 10000

        # Binary exponentiation
        result = 1
        b = base % MOD
        e = exp
        while e > 0:
            if e & 1:
                result = (result * b) % MOD
            b = (b * b) % MOD
            e >>= 1

        total += result

    return int(total)
