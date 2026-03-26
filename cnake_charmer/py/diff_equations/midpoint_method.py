"""Solve y'=y*cos(t) from t=0 to t=5 using midpoint method with n steps.

Initial condition y(0) = 1. Returns the final value of y.

Keywords: ODE, midpoint method, differential equation, numerical, second-order
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(5000000,))
def midpoint_method(n: int) -> float:
    """Solve y'=y*cos(t) using midpoint method and return final y value.

    Args:
        n: Number of integration steps.

    Returns:
        Final value of y at t=5.
    """
    t = 0.0
    y = 1.0
    dt = 5.0 / n

    for _ in range(n):
        # Midpoint method: use Euler to estimate midpoint, then use midpoint slope
        k1 = y * math.cos(t)
        y_mid = y + 0.5 * dt * k1
        t_mid = t + 0.5 * dt
        k2 = y_mid * math.cos(t_mid)
        y += dt * k2
        t += dt

    return y
