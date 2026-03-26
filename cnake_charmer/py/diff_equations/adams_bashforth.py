"""Solve y'=-2*y+t using 4-step Adams-Bashforth method with n steps.

From t=0 to t=5, y(0)=1. Bootstrap with RK4. Returns final y.

Keywords: ODE, Adams-Bashforth, multistep method, differential equation, numerical
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(5000000,))
def adams_bashforth(n: int) -> float:
    """Solve y'=-2*y+t using 4-step Adams-Bashforth and return final y.

    Args:
        n: Number of integration steps.

    Returns:
        Final value of y at t=5.
    """
    dt = 5.0 / n

    def f(t_val, y_val):
        return -2.0 * y_val + t_val

    # Bootstrap first 4 steps with RK4
    t_hist = [0.0] * 4
    y_hist = [0.0] * 4
    f_hist = [0.0] * 4

    t_hist[0] = 0.0
    y_hist[0] = 1.0
    f_hist[0] = f(t_hist[0], y_hist[0])

    for i in range(1, 4):
        t_val = t_hist[i - 1]
        y_val = y_hist[i - 1]
        k1 = dt * f(t_val, y_val)
        k2 = dt * f(t_val + 0.5 * dt, y_val + 0.5 * k1)
        k3 = dt * f(t_val + 0.5 * dt, y_val + 0.5 * k2)
        k4 = dt * f(t_val + dt, y_val + k3)
        y_hist[i] = y_val + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        t_hist[i] = t_val + dt
        f_hist[i] = f(t_hist[i], y_hist[i])

    # Adams-Bashforth 4-step: y_{n+1} = y_n + dt/24*(55*f_n - 59*f_{n-1} + 37*f_{n-2} - 9*f_{n-3})
    t_val = t_hist[3]
    y_val = y_hist[3]
    fm3 = f_hist[0]
    fm2 = f_hist[1]
    fm1 = f_hist[2]
    fm0 = f_hist[3]

    for _i in range(4, n):
        y_val = y_val + dt / 24.0 * (55.0 * fm0 - 59.0 * fm1 + 37.0 * fm2 - 9.0 * fm3)
        t_val += dt
        fm3 = fm2
        fm2 = fm1
        fm1 = fm0
        fm0 = f(t_val, y_val)

    return y_val
