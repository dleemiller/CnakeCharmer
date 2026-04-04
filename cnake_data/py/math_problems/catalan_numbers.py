"""Compute first n Catalan numbers using Pascal's triangle for binomial coefficients.

Keywords: catalan, combinatorics, binomial, pascal, modular arithmetic, benchmark
"""

from cnake_data.benchmarks import python_benchmark

MOD = 10**9 + 7


@python_benchmark(args=(100000,))
def catalan_numbers(n: int) -> int:
    """Compute sum of the first n Catalan numbers mod 10^9+7.

    C(k) = binomial(2k, k) / (k+1). Uses Pascal's triangle to build
    binomial coefficients row by row, then computes each Catalan number
    using modular inverse via Fermat's little theorem.

    Args:
        n: Number of Catalan numbers to compute.

    Returns:
        Sum of C(0) through C(n-1), mod 10^9+7.
    """
    mod = MOD

    # Precompute factorials and inverse factorials mod p
    max_val = 2 * n + 1
    fact = [1] * (max_val + 1)
    for i in range(1, max_val + 1):
        fact[i] = fact[i - 1] * i % mod

    inv_fact = [1] * (max_val + 1)
    inv_fact[max_val] = pow(fact[max_val], mod - 2, mod)
    for i in range(max_val - 1, -1, -1):
        inv_fact[i] = inv_fact[i + 1] * (i + 1) % mod

    total = 0
    for k in range(n):
        # C(k) = binom(2k, k) / (k+1)
        binom = fact[2 * k] * inv_fact[k] % mod * inv_fact[k] % mod
        cat = binom * inv_fact[k + 1] % mod * fact[k] % mod
        total = (total + cat) % mod

    return total
