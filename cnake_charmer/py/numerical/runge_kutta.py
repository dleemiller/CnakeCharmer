"""Solve y'=y*sin(t) from t=0 to t=10 using RK4 with n steps.

Initial condition y(0) = 1. Returns the final value of y.

Keywords: numerical, ODE, Runge-Kutta, RK4, differential equation, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def runge_kutta(n: int) -> float:
    """Solve y'=y*sin(t) using RK4 and return final y value.

    Args:
        n: Number of integration steps.

    Returns:
        Final value of y at t=10.
    """
    t = 0.0
    y = 1.0
    dt = 10.0 / n

    for _ in range(n):
        k1 = dt * y * math.sin(t)
        k2 = dt * (y + 0.5 * k1) * math.sin(t + 0.5 * dt)
        k3 = dt * (y + 0.5 * k2) * math.sin(t + 0.5 * dt)
        k4 = dt * (y + k3) * math.sin(t + dt)
        y += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        t += dt

    return y
