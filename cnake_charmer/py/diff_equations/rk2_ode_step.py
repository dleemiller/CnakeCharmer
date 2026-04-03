"""Solve du/dt = -u + sin(t) from t=0 to t=10 using RK2 midpoint method with n steps.

Initial condition u(0) = 0. Returns tuple of (final_u, u_at_midpoint).
Uses Taylor series approximation for sin to keep computation in pure Python loops.

Keywords: ODE, RK2, Runge-Kutta, midpoint method, differential equation, numerical
"""

from cnake_charmer.benchmarks import python_benchmark


def _taylor_sin(x: float) -> float:
    """Approximate sin(x) using Taylor series with range reduction."""
    # Reduce x to [-pi, pi]
    pi = 3.141592653589793
    two_pi = 6.283185307179586
    x = x - two_pi * int(x / two_pi + (0.5 if x >= 0 else -0.5))
    # Reduce to [-pi/2, pi/2] using sin(pi - x) = sin(x)
    if x > 1.5707963267948966:
        x = pi - x
    elif x < -1.5707963267948966:
        x = -pi - x
    x2 = x * x
    # Horner form for Taylor series up to x^19
    return x * (
        1.0
        + x2
        * (
            -1.0 / 6.0
            + x2
            * (
                1.0 / 120.0
                + x2
                * (
                    -1.0 / 5040.0
                    + x2
                    * (
                        1.0 / 362880.0
                        + x2
                        * (
                            -1.0 / 39916800.0
                            + x2
                            * (
                                1.0 / 6227020800.0
                                + x2
                                * (
                                    -1.0 / 1307674368000.0
                                    + x2
                                    * (1.0 / 355687428096000.0 + x2 * (-1.0 / 121645100408832000.0))
                                )
                            )
                        )
                    )
                )
            )
        )
    )


@python_benchmark(args=(200000,))
def rk2_ode_step(n: int) -> tuple:
    """Solve du/dt = -u + sin(t) using RK2 midpoint method.

    Args:
        n: Number of integration steps.

    Returns:
        Tuple of (final_u, u_at_midpoint).
    """
    u = 0.0
    t = 0.0
    dt = 10.0 / n
    mid_step = n // 2
    u_at_mid = 0.0

    for i in range(n):
        # f(u, t) = -u + sin(t)
        f1 = -u + _taylor_sin(t)
        K1 = dt * f1
        # f(u + 0.5*K1, t + 0.5*dt)
        f2 = -(u + 0.5 * K1) + _taylor_sin(t + 0.5 * dt)
        K2 = dt * f2
        u = u + K2
        t += dt

        if i == mid_step:
            u_at_mid = u

    return (u, u_at_mid)
