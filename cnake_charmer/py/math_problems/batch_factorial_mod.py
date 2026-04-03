"""
Batch factorial modular arithmetic: compute k! mod p for k=1..n.

Keywords: math, factorial, modular arithmetic, number theory, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def batch_factorial_mod(n: int) -> tuple:
    """Compute k! mod p for each k from 1 to n and return summary statistics.

    Uses iterative multiplication to build factorials incrementally.
    Returns (sum_of_factorials_mod_p, xor_of_factorials_mod_p) where
    each k! is first reduced mod p before accumulation.

    Args:
        n: Upper bound (inclusive) for factorial computation.

    Returns:
        Tuple of (sum_mod, xor_accum).
    """
    MOD = 1000000007
    total = 0
    xor_accum = 0
    fact = 1
    for k in range(1, n + 1):
        fact = (fact * k) % MOD
        total = (total + fact) % MOD
        xor_accum = xor_accum ^ fact
    return (total, xor_accum)
