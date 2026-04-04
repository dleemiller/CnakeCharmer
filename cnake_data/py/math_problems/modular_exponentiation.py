"""
Sum of modular exponentiations using binary exponentiation.

Keywords: math, modular exponentiation, binary exponentiation, number theory, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def modular_exponentiation(n: int) -> int:
    """Compute sum of (base^exp) mod p for n pairs.

    For each i: base = (i*7+3) % 1000, exp = (i*13+7) % 10000, p = 10^9+7.
    Uses fast binary exponentiation (repeated squaring).

    Args:
        n: Number of pairs to compute.

    Returns:
        Sum of all results as an integer.
    """
    MOD = 1000000007
    total = 0

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

    return total
