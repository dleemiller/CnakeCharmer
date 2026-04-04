"""Sum Fibonacci numbers using an iterator.

Keywords: fibonacci, iterator, generator, math, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def fibonacci_iterator(n: int) -> int:
    """Generate n Fibonacci numbers (mod 10^9+7) and compute their weighted sum.

    Args:
        n: Number of Fibonacci numbers to generate.

    Returns:
        Weighted sum: sum(fib_i * (i % 256 + 1)) mod 10^9+7.
    """
    MOD = 1000000007
    a = 0
    b = 1
    total = 0
    for i in range(n):
        total = (total + a * ((i % 256) + 1)) % MOD
        a, b = b, (a + b) % MOD

    return total
