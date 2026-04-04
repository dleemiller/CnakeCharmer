"""Compute midpoint Riemann sum of f(x) = sin(x)/x (sinc) from 0.001 to 10*pi.

Keywords: numerical integration, Riemann sum, midpoint, sinc function, quadrature
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(10000000,))
def riemann_sum_midpoint(n: int) -> float:
    """Compute midpoint Riemann sum of sin(x)/x from 0.001 to 10*pi.

    Args:
        n: Number of rectangles.

    Returns:
        Approximate integral value.
    """
    a = 0.001
    b = 10.0 * math.pi
    dx = (b - a) / n
    total = 0.0

    for i in range(n):
        x = a + (i + 0.5) * dx
        total += math.sin(x) / x

    return total * dx
