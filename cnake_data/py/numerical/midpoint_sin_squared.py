"""Midpoint rule integration of sin^2(x) over [0, pi].

Numerical integration using the midpoint (rectangle) rule, evaluating
sin^2 at the center of each subinterval.

Keywords: numerical, integration, midpoint, trigonometry, sin_squared, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(5000000,))
def midpoint_sin_squared(n: int) -> tuple:
    """Integrate sin^2(x) from 0 to pi using the midpoint rule with n steps.

    Also computes the integral using n//2 steps for Richardson-style
    error estimation.

    Args:
        n: Number of subintervals for the midpoint rule.

    Returns:
        Tuple of (integral_n, integral_half_n).
    """
    pi = math.pi
    sin = math.sin

    # Full resolution: n steps
    dx = pi / n
    total = 0.0
    for i in range(n):
        x = (i + 0.5) * dx
        s = sin(x)
        total += s * s
    integral_n = total * dx

    # Half resolution: n//2 steps
    half_n = n // 2
    if half_n < 1:
        half_n = 1
    dx2 = pi / half_n
    total2 = 0.0
    for i in range(half_n):
        x = (i + 0.5) * dx2
        s = sin(x)
        total2 += s * s
    integral_half_n = total2 * dx2

    return (integral_n, integral_half_n)
