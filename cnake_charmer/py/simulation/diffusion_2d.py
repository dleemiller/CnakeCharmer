"""2D diffusion simulation on a grid using Laplacian stencil.

Keywords: simulation, diffusion, 2D, Laplacian, PDE, finite difference, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(200,))
def diffusion_2d(n: int) -> float:
    """Simulate 2D diffusion on an n x n grid for 100 timesteps.

    Initial condition: u[i][j] = 1.0 if i==n//2 and j==n//2 else 0.0
    Laplacian stencil with alpha=0.1.
    Returns sum of final grid values.

    Args:
        n: Grid dimension (n x n).

    Returns:
        Sum of all grid values after simulation.
    """
    timesteps = 100
    alpha = 0.1

    # Flat arrays for n x n grid
    u = [0.0] * (n * n)
    u_new = [0.0] * (n * n)
    u[n * (n // 2) + n // 2] = 1.0

    for _ in range(timesteps):
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                idx = i * n + j
                laplacian = (
                    u[(i - 1) * n + j]
                    + u[(i + 1) * n + j]
                    + u[i * n + j - 1]
                    + u[i * n + j + 1]
                    - 4.0 * u[idx]
                )
                u_new[idx] = u[idx] + alpha * laplacian
        # Swap
        u, u_new = u_new, u
        # Zero the boundary in u_new for next iteration
        for i in range(n):
            u_new[i] = 0.0
            u_new[(n - 1) * n + i] = 0.0
            u_new[i * n] = 0.0
            u_new[i * n + n - 1] = 0.0

    total = 0.0
    for i in range(n * n):
        total += u[i]
    return total
