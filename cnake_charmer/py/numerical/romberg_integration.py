"""Romberg integration of sin(x).

Keywords: numerical, integration, romberg, richardson extrapolation, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(25,))
def romberg_integration(n: int) -> float:
    """Compute Romberg integration of sin(x) from 0 to pi with n levels.

    Uses Richardson extrapolation on the trapezoidal rule to achieve
    high-accuracy numerical integration.

    Args:
        n: Number of Romberg levels (tableau size).

    Returns:
        The computed integral value.
    """
    a = 0.0
    b = math.pi

    # R[i][j] is the Romberg tableau
    R = [[0.0] * n for _ in range(n)]

    # R[0][0] = basic trapezoidal rule
    R[0][0] = 0.5 * (b - a) * (math.sin(a) + math.sin(b))

    for i in range(1, n):
        # Composite trapezoidal rule with 2^i intervals
        h = (b - a) / (1 << i)

        # Add new midpoints
        total = 0.0
        for k in range(1, (1 << i), 2):
            total += math.sin(a + k * h)

        R[i][0] = 0.5 * R[i - 1][0] + h * total

        # Richardson extrapolation
        for j in range(1, i + 1):
            factor = 4**j
            R[i][j] = (factor * R[i][j - 1] - R[i - 1][j - 1]) / (factor - 1)

    return R[n - 1][n - 1]
