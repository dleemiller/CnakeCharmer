# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute first n Catalan numbers using modular arithmetic (Cython-optimized).

Keywords: catalan, combinatorics, binomial, modular arithmetic, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark

DEF MOD = 1000000007


@cython_benchmark(syntax="cy", args=(100000,))
def catalan_numbers(int n):
    """Compute sum of the first n Catalan numbers mod 10^9+7.

    Uses precomputed factorials and inverse factorials with C arrays.

    Args:
        n: Number of Catalan numbers to compute.

    Returns:
        Sum of C(0) through C(n-1), mod 10^9+7.
    """
    cdef int i, k
    cdef long long binom, cat, total
    cdef int max_val = 2 * n + 1

    cdef long long *fact = <long long *>malloc((max_val + 1) * sizeof(long long))
    cdef long long *inv_fact = <long long *>malloc((max_val + 1) * sizeof(long long))
    if fact == NULL or inv_fact == NULL:
        if fact != NULL: free(fact)
        if inv_fact != NULL: free(inv_fact)
        raise MemoryError()

    # Precompute factorials
    fact[0] = 1
    for i in range(1, max_val + 1):
        fact[i] = fact[i - 1] * i % MOD

    # Precompute inverse factorials using Fermat's little theorem
    # inv_fact[max_val] = pow(fact[max_val], MOD-2, MOD)
    cdef long long base, exp_val, result
    base = fact[max_val]
    exp_val = MOD - 2
    result = 1
    while exp_val > 0:
        if exp_val & 1:
            result = result * base % MOD
        base = base * base % MOD
        exp_val >>= 1
    inv_fact[max_val] = result

    for i in range(max_val - 1, -1, -1):
        inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD

    total = 0
    for k in range(n):
        # C(k) = binom(2k, k) / (k+1) = binom(2k,k) * inv_fact[k+1] * fact[k]
        binom = fact[2 * k] * inv_fact[k] % MOD * inv_fact[k] % MOD
        cat = binom * inv_fact[k + 1] % MOD * fact[k] % MOD
        total = (total + cat) % MOD

    free(fact)
    free(inv_fact)
    return int(total)
