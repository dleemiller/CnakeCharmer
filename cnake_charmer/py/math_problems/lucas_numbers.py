"""Compute sum of first n Lucas numbers mod 10^9+7.

Keywords: lucas, fibonacci, sequence, number theory, modular arithmetic, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(10000000,))
def lucas_numbers(n: int) -> int:
    """Compute the sum of the first n Lucas numbers mod 10^9+7.

    Lucas numbers: L(0)=2, L(1)=1, L(n)=L(n-1)+L(n-2).
    Returns sum(L(0)..L(n-1)) mod 10^9+7.

    Args:
        n: How many Lucas numbers to sum.

    Returns:
        Sum of first n Lucas numbers mod 10^9+7.
    """
    MOD = 1000000007
    if n == 0:
        return 0
    if n == 1:
        return 2

    prev2 = 2  # L(0)
    prev1 = 1  # L(1)
    total = prev2 + prev1

    for _ in range(2, n):
        curr = (prev1 + prev2) % MOD
        total = (total + curr) % MOD
        prev2 = prev1
        prev1 = curr

    return total % MOD
