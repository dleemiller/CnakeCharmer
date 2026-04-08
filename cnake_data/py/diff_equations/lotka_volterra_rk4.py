"""Lotka-Volterra three-species predator-prey ODE solved with RK4.

Integrates a competitive three-species system using the classical
fourth-order Runge-Kutta method with a fixed time step h=0.1.

Keywords: lotka-volterra, predator-prey, runge-kutta, ODE, numerical integration
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(2000,))
def lotka_volterra_rk4(n_steps: int) -> tuple:
    """Integrate a three-species Lotka-Volterra system with RK4.

    Args:
        n_steps: Number of integration steps.

    Returns:
        Tuple of (mean_s, mean_z, mean_r) — time-averaged values over the trajectory.
    """
    # Fixed system parameters
    alpha = 1.0  # prey birth rate
    beta = 0.1  # predation rate
    delta = 0.05  # prey death rate (non-predation)
    xi = 0.075  # predator growth from predation
    alpha2 = 0.04  # top-predator birth from meso-predator
    sigma = 0.2  # meso-predator death

    h = 0.1
    s = 10.0  # prey
    z = 5.0  # meso-predator
    r = 2.0  # top-predator

    sum_s = 0.0
    sum_z = 0.0
    sum_r = 0.0

    for _ in range(n_steps):

        def ds(s_, z_, r_):
            return alpha * s_ - beta * s_ * z_ - delta * s_

        def dz(s_, z_, r_):
            return beta * s_ * z_ + xi * r_ - alpha2 * s_ * z_ - sigma * z_

        def dr(s_, z_, r_):
            return delta * s_ + alpha2 * s_ * z_ - xi * r_ + sigma * z_

        k1 = h * ds(s, z, r)
        l1 = h * dz(s, z, r)
        m1 = h * dr(s, z, r)

        k2 = h * ds(s + k1 / 2, z + l1 / 2, r + m1 / 2)
        l2 = h * dz(s + k1 / 2, z + l1 / 2, r + m1 / 2)
        m2 = h * dr(s + k1 / 2, z + l1 / 2, r + m1 / 2)

        k3 = h * ds(s + k2 / 2, z + l2 / 2, r + m2 / 2)
        l3 = h * dz(s + k2 / 2, z + l2 / 2, r + m2 / 2)
        m3 = h * dr(s + k2 / 2, z + l2 / 2, r + m2 / 2)

        k4 = h * ds(s + k3, z + l3, r + m3)
        l4 = h * dz(s + k3, z + l3, r + m3)
        m4 = h * dr(s + k3, z + l3, r + m3)

        s = max(0.0, s + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0)
        z = max(0.0, z + (l1 + 2 * l2 + 2 * l3 + l4) / 6.0)
        r = max(0.0, r + (m1 + 2 * m2 + 2 * m3 + m4) / 6.0)

        sum_s += s
        sum_z += z
        sum_r += r

    return (sum_s / n_steps, sum_z / n_steps, sum_r / n_steps)
