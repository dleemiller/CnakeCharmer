"""Babylonian method for computing square roots.

Iteratively approximates sqrt(value) using the recurrence x = (x + value/x) / 2
until convergence within a tolerance.

Keywords: sqrt, babylonian, newton, numerical, iterative, convergence
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def babylonian_sqrt_sum(n: int) -> float:
    """Compute sqrt via Babylonian method for integers 1..n and return their sum.

    Args:
        n: Number of square roots to compute.

    Returns:
        Sum of sqrt(1) + sqrt(2) + ... + sqrt(n) computed via Babylonian method.
    """
    total = 0.0
    for val in range(1, n + 1):
        x = 1.0
        for _ in range(100):
            x_prev = x
            x = (x + val / x) / 2.0
            if abs(x_prev - x) < 1e-14:
                break
        total += x
    return total
