"""1D wave equation simulation using finite differences.

Keywords: wave equation, simulation, PDE, finite difference, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(5000,))
def wave_equation(n: int) -> float:
    """Simulate 1D wave equation on n cells for 500 timesteps.

    Initial condition: u[i] = sin(i * pi / n) * 100.
    Wave speed c=1.0, dt=0.01, dx=1.0/n.
    Boundary conditions: u[0] = u[n-1] = 0.

    Args:
        n: Number of spatial cells.

    Returns:
        Sum of final amplitudes.
    """
    timesteps = 500
    c = 1.0
    dt = 0.01
    dx = 1.0 / n
    r = (c * dt / dx) ** 2
    pi = math.pi

    # Current and previous timestep arrays
    u = [math.sin(i * pi / n) * 100.0 for i in range(n)]
    u_prev = [math.sin(i * pi / n) * 100.0 for i in range(n)]
    u_next = [0.0] * n

    u[0] = 0.0
    u[n - 1] = 0.0
    u_prev[0] = 0.0
    u_prev[n - 1] = 0.0

    for _t in range(timesteps):
        u_next[0] = 0.0
        u_next[n - 1] = 0.0
        for i in range(1, n - 1):
            u_next[i] = 2.0 * u[i] - u_prev[i] + r * (u[i - 1] - 2.0 * u[i] + u[i + 1])
        u_prev, u, u_next = u, u_next, u_prev

    total = 0.0
    for i in range(n):
        total += u[i]
    return total
