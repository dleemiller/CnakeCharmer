"""Sum of climbing stairs ways for 1..n using Fibonacci recurrence.

Keywords: leetcode, climbing stairs, fibonacci, dynamic programming, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(3000000,))
def climbing_stairs(n: int) -> int:
    """Compute sum of ways(1) + ways(2) + ... + ways(n) mod 10^9+7.

    ways(k) = fib(k+1) where fib is the Fibonacci sequence.

    Args:
        n: Upper limit.

    Returns:
        Sum modulo 10^9 + 7.
    """
    MOD = 1000000007
    total = 0
    a, b = 1, 1  # fib(1)=1, fib(2)=1
    for _ in range(n):
        total = (total + b) % MOD
        a, b = b, (a + b) % MOD
    return total
