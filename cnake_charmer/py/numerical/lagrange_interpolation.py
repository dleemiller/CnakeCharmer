"""Lagrange polynomial interpolation.

Keywords: numerical, interpolation, lagrange, polynomial, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(2000,))
def lagrange_interpolation(n: int) -> float:
    """Interpolate at x=0.5 using n Lagrange basis polynomials.

    Points: x_i = i/(n-1), y_i = sin(x_i * pi).

    Args:
        n: Number of interpolation points.

    Returns:
        Interpolated value at x=0.5.
    """
    if n < 2:
        return 0.0

    target = 0.5
    result = 0.0

    for i in range(n):
        xi = i / (n - 1)
        yi = math.sin(xi * math.pi)

        basis = 1.0
        for j in range(n):
            if j != i:
                xj = j / (n - 1)
                basis *= (target - xj) / (xi - xj)

        result += yi * basis

    return result
