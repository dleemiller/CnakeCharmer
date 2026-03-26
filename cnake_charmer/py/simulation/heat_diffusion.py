"""1D heat equation simulation using finite differences.

Keywords: heat equation, diffusion, finite difference, simulation, PDE, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000,))
def heat_diffusion(n: int) -> float:
    """Simulate 1D heat equation using finite differences.

    Simulates n cells over 1000 timesteps with diffusivity alpha=0.1.
    Initial condition: u[i] = sin(i * pi / n) * 100.
    Boundary conditions: u[0] = u[n-1] = 0.

    Args:
        n: Number of cells.

    Returns:
        Sum of final temperature values.
    """
    timesteps = 1000
    alpha = 0.1
    pi = math.pi

    # Initialize
    u = [math.sin(i * pi / n) * 100.0 for i in range(n)]
    u[0] = 0.0
    u[n - 1] = 0.0

    u_new = [0.0] * n

    for _t in range(timesteps):
        u_new[0] = 0.0
        u_new[n - 1] = 0.0
        for i in range(1, n - 1):
            u_new[i] = u[i] + alpha * (u[i - 1] - 2.0 * u[i] + u[i + 1])
        u, u_new = u_new, u

    total = 0.0
    for i in range(n):
        total += u[i]
    return total
