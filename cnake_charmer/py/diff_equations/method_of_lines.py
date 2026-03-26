"""Solve 1D advection equation du/dt + du/dx = 0 using method of lines.

Spatial finite differences + RK4 in time on n cells for 1000 steps.
u(x,0) = exp(-((x-0.5)/0.1)^2). Return sum of final u.

Keywords: PDE, advection, method of lines, RK4, finite difference, numerical
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(5000,))
def method_of_lines(n: int) -> float:
    """Solve 1D advection equation using method of lines.

    Args:
        n: Number of spatial cells.

    Returns:
        Sum of final u values.
    """
    num_steps = 1000
    dx = 1.0 / n
    dt = 0.5 * dx  # CFL condition

    # Initial condition
    u = [0.0] * n
    for i in range(n):
        x = i * dx
        arg = (x - 0.5) / 0.1
        u[i] = math.exp(-arg * arg)

    # Temporary arrays for RK4
    k1 = [0.0] * n
    k2 = [0.0] * n
    k3 = [0.0] * n
    k4 = [0.0] * n
    tmp = [0.0] * n

    def compute_rhs(src, dst):
        """Compute spatial derivative using upwind scheme: -du/dx."""
        dst[0] = -(src[0] - src[n - 1]) / dx
        for i in range(1, n):
            dst[i] = -(src[i] - src[i - 1]) / dx

    for _ in range(num_steps):
        # k1
        compute_rhs(u, k1)

        # k2
        for i in range(n):
            tmp[i] = u[i] + 0.5 * dt * k1[i]
        compute_rhs(tmp, k2)

        # k3
        for i in range(n):
            tmp[i] = u[i] + 0.5 * dt * k2[i]
        compute_rhs(tmp, k3)

        # k4
        for i in range(n):
            tmp[i] = u[i] + dt * k3[i]
        compute_rhs(tmp, k4)

        # Update
        for i in range(n):
            u[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])

    total = 0.0
    for i in range(n):
        total += u[i]

    return total
