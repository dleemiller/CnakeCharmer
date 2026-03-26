"""Solve y'=-y+sin(t) from t=0 to t=10 using forward Euler with n steps.

Initial condition y(0) = 1. Returns the final value of y.

Keywords: ODE, Euler method, differential equation, forward Euler, numerical
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(10000000,))
def euler_method(n: int) -> float:
    """Solve y'=-y+sin(t) using forward Euler and return final y value.

    Args:
        n: Number of integration steps.

    Returns:
        Final value of y at t=10.
    """
    t = 0.0
    y = 1.0
    dt = 10.0 / n

    for _ in range(n):
        y += dt * (-y + math.sin(t))
        t += dt

    return y
