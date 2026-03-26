"""Compute sum of Fibonacci numbers F(1)+F(2)+...+F(n) using iterative recurrence.

Keywords: fibonacci, modular arithmetic, number theory, accumulation, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark

MOD = 10**9 + 7


@python_benchmark(args=(10000000,))
def fibonacci_matrix(n: int) -> int:
    """Compute sum F(1)+F(2)+...+F(n) mod 10^9+7.

    Uses iterative Fibonacci recurrence and accumulates the sum.

    Args:
        n: Upper limit of Fibonacci sum.

    Returns:
        Sum of first n Fibonacci numbers, mod 10^9+7.
    """
    mod = MOD
    a = 0
    b = 1
    total = 0
    for _ in range(n):
        a, b = b, (a + b) % mod
        total = (total + a) % mod
    return total
