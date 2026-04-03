# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Batch factorial modular arithmetic: compute k! mod p for k=1..n.

Keywords: math, factorial, modular arithmetic, number theory, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def batch_factorial_mod(int n):
    """Compute k! mod p for each k from 1 to n and return summary statistics."""
    cdef long long MOD = 1000000007
    cdef long long total = 0
    cdef long long xor_accum = 0
    cdef long long fact = 1
    cdef int k

    with nogil:
        for k in range(1, n + 1):
            fact = (fact * k) % MOD
            total = (total + fact) % MOD
            xor_accum = xor_accum ^ fact

    return (total, xor_accum)
