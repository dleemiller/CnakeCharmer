"""Solve y'=y*cos(t) from t=0 to t=5 using midpoint method with n steps.

Initial condition y(0) = 1. Returns tuple of trajectory metrics.

Keywords: ODE, midpoint method, differential equation, numerical, second-order
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(5000000,))
def midpoint_method(n: int) -> tuple:
    """Solve y'=y*cos(t) using midpoint method and return trajectory metrics.

    Args:
        n: Number of integration steps.

    Returns:
        Tuple of (final_y, y_at_mid, max_y).
    """
    t = 0.0
    y = 1.0
    dt = 5.0 / n
    mid_step = n // 2
    max_y = y
    y_at_mid = y

    for i in range(n):
        # Midpoint method: use Euler to estimate midpoint, then use midpoint slope
        k1 = y * math.cos(t)
        y_mid = y + 0.5 * dt * k1
        t_mid = t + 0.5 * dt
        k2 = y_mid * math.cos(t_mid)
        y += dt * k2
        t += dt

        if y > max_y:
            max_y = y
        if i == mid_step:
            y_at_mid = y

    return (y, y_at_mid, max_y)
