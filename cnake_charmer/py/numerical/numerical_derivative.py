"""Compute numerical derivative using central differences.

Keywords: numerical derivative, central difference, differentiation, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def numerical_derivative(n: int) -> float:
    """Compute numerical derivative of f(x)=sin(x)*exp(-x/100) at n points.

    Points are evenly spaced from 0 to 10. Uses central differences with
    step h = 10/(n-1).

    Args:
        n: Number of evaluation points.

    Returns:
        Sum of all derivative values as a float.
    """
    if n < 3:
        return 0.0

    dx = 10.0 / (n - 1)
    h = dx

    total = 0.0
    for i in range(1, n - 1):
        x_plus = (i + 1) * dx
        x_minus = (i - 1) * dx
        f_plus = math.sin(x_plus) * math.exp(-x_plus / 100.0)
        f_minus = math.sin(x_minus) * math.exp(-x_minus / 100.0)
        deriv = (f_plus - f_minus) / (2.0 * h)
        total += deriv

    return total
