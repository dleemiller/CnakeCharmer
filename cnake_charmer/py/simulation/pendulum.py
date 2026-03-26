"""Double pendulum simulation using RK4 integration.

Keywords: simulation, double pendulum, RK4, physics, ODE, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(200000,))
def pendulum(n: int) -> float:
    """Simulate a double pendulum using RK4 for n timesteps.

    Parameters: m1=m2=1.0, L1=L2=1.0, g=9.81, dt=0.001.
    Initial: theta1=pi/4, theta2=pi/2, omega1=omega2=0.

    The equations of motion for a double pendulum are integrated
    using the 4th-order Runge-Kutta method.

    Args:
        n: Number of timesteps.

    Returns:
        Final value of theta1.
    """
    dt = 0.001
    g = 9.81
    m1 = 1.0
    m2 = 1.0
    L1 = 1.0
    L2 = 1.0
    pi = math.pi

    th1 = pi / 4.0
    th2 = pi / 2.0
    w1 = 0.0
    w2 = 0.0

    def derivs(th1, w1, th2, w2):
        delta = th2 - th1
        sin_d = math.sin(delta)
        cos_d = math.cos(delta)
        den1 = (m1 + m2) * L1 - m2 * L1 * cos_d * cos_d
        den2 = (L2 / L1) * den1

        dw1 = (
            m2 * L1 * w1 * w1 * sin_d * cos_d
            + m2 * g * math.sin(th2) * cos_d
            + m2 * L2 * w2 * w2 * sin_d
            - (m1 + m2) * g * math.sin(th1)
        ) / den1
        dw2 = (
            -m2 * L2 * w2 * w2 * sin_d * cos_d
            + (m1 + m2) * g * math.sin(th1) * cos_d
            - (m1 + m2) * L1 * w1 * w1 * sin_d
            - (m1 + m2) * g * math.sin(th2)
        ) / den2
        return w1, dw1, w2, dw2

    for _ in range(n):
        k1_th1, k1_w1, k1_th2, k1_w2 = derivs(th1, w1, th2, w2)
        k2_th1, k2_w1, k2_th2, k2_w2 = derivs(
            th1 + 0.5 * dt * k1_th1,
            w1 + 0.5 * dt * k1_w1,
            th2 + 0.5 * dt * k1_th2,
            w2 + 0.5 * dt * k1_w2,
        )
        k3_th1, k3_w1, k3_th2, k3_w2 = derivs(
            th1 + 0.5 * dt * k2_th1,
            w1 + 0.5 * dt * k2_w1,
            th2 + 0.5 * dt * k2_th2,
            w2 + 0.5 * dt * k2_w2,
        )
        k4_th1, k4_w1, k4_th2, k4_w2 = derivs(
            th1 + dt * k3_th1, w1 + dt * k3_w1, th2 + dt * k3_th2, w2 + dt * k3_w2
        )

        th1 += dt * (k1_th1 + 2.0 * k2_th1 + 2.0 * k3_th1 + k4_th1) / 6.0
        w1 += dt * (k1_w1 + 2.0 * k2_w1 + 2.0 * k3_w1 + k4_w1) / 6.0
        th2 += dt * (k1_th2 + 2.0 * k2_th2 + 2.0 * k3_th2 + k4_th2) / 6.0
        w2 += dt * (k1_w2 + 2.0 * k2_w2 + 2.0 * k3_w2 + k4_w2) / 6.0

    return th1
