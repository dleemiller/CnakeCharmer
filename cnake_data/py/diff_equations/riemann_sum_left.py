"""Compute left Riemann sum of f(x) = x^2 * exp(-x) from 0 to 10 with n rectangles.

Keywords: numerical integration, Riemann sum, left endpoint, quadrature
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(10000000,))
def riemann_sum_left(n: int) -> float:
    """Compute left Riemann sum of x^2 * exp(-x) from 0 to 10.

    Args:
        n: Number of rectangles.

    Returns:
        Approximate integral value.
    """
    dx = 10.0 / n
    total = 0.0

    for i in range(n):
        x = i * dx
        total += x * x * math.exp(-x)

    return total * dx
