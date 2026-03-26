"""Fibonacci sequence generator.

Keywords: fibonacci, algorithms, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def fib(n: int) -> int:
    """Compute sum of first n Fibonacci numbers modulo 10^9+7.

    Args:
        n: How many Fibonacci numbers to compute.

    Returns:
        Sum of F(1) + F(2) + ... + F(n) mod 10^9+7.
    """
    mod = 1000000007
    a, b = 0, 1
    total = 0
    for _ in range(n):
        total = (total + b) % mod
        a, b = b, (a + b) % mod
    return total
