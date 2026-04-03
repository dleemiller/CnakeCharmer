"""Midpoint-rule numerical integration of multiple transcendental functions.

Keywords: numerical integration, midpoint rule, sin, exp, transcendental
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(0.0, 10.0, 200000))
def midpoint_integrate(start, stop, n):
    """Integrate sin(x)^2, exp(-x^2), and x*sin(x) over [start, stop] using midpoint rule.

    Args:
        start: Lower integration bound.
        stop: Upper integration bound.
        n: Number of subintervals.

    Returns:
        Tuple of (integral_sin_sq, integral_gaussian, integral_xsinx).
    """
    dx = (stop - start) / n

    sum_sin_sq = 0.0
    sum_gauss = 0.0
    sum_xsinx = 0.0

    for i in range(n):
        x = start + (i + 0.5) * dx
        s = math.sin(x)
        sum_sin_sq += s * s
        sum_gauss += math.exp(-x * x)
        sum_xsinx += x * s

    integral_sin_sq = sum_sin_sq * dx
    integral_gauss = sum_gauss * dx
    integral_xsinx = sum_xsinx * dx

    return (integral_sin_sq, integral_gauss, integral_xsinx)
