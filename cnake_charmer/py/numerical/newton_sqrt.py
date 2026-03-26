"""Compute sum of square roots of 1..n using Newton's method.

Keywords: numerical, Newton's method, square root, iteration, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def newton_sqrt(n: int) -> float:
    """Compute the sum of square roots of integers 1 through n using Newton's method.

    Each square root is approximated with 5 iterations of Newton's method
    starting from x/2.

    Args:
        n: Upper bound of range (inclusive).

    Returns:
        Sum of approximate square roots.
    """
    total = 0.0
    for k in range(1, n + 1):
        x = k * 0.5  # initial guess
        for _ in range(5):
            x = 0.5 * (x + k / x)
        total += x
    return total
