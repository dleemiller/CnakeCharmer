"""1D fluid simulation using Burgers' equation with upwind scheme.

Keywords: simulation, fluid, burgers equation, upwind, PDE, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(5000,))
def fluid_1d(n: int) -> float:
    """Simulate 1D Burgers' equation on n cells for 500 steps.

    Initial condition: u[i] = sin(2*pi*i/n).
    Uses first-order upwind finite difference scheme with dt=0.001, dx=1/n.
    Periodic boundary conditions.

    Args:
        n: Number of cells.

    Returns:
        Sum of final velocity values.
    """
    steps = 500
    dt = 0.001
    dx = 1.0 / n
    pi = math.pi

    u = [math.sin(2.0 * pi * i / n) for i in range(n)]
    u_new = [0.0] * n

    for _t in range(steps):
        for i in range(n):
            du = u[i] - u[(i - 1) % n] if u[i] >= 0 else u[(i + 1) % n] - u[i]
            u_new[i] = u[i] - dt * u[i] * du / dx
        u, u_new = u_new, u

    total = 0.0
    for i in range(n):
        total += u[i]
    return total
