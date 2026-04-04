"""Solve 2D Laplace equation on n x n grid using Jacobi iteration (500 iterations).

Boundary conditions: top=100, others=0. Returns sum of interior values.

Keywords: PDE, Laplace equation, Jacobi iteration, finite difference, numerical
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100,))
def finite_difference_laplacian(n: int) -> float:
    """Solve 2D Laplace equation using Jacobi iteration.

    Args:
        n: Grid size (n x n).

    Returns:
        Sum of interior grid values after 500 iterations.
    """
    num_iters = 500

    # Initialize grid (flat array, row-major)
    grid = [0.0] * (n * n)
    new_grid = [0.0] * (n * n)

    # Set top boundary to 100
    for j in range(n):
        grid[j] = 100.0
        new_grid[j] = 100.0

    # Jacobi iteration
    for _ in range(num_iters):
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                new_grid[i * n + j] = 0.25 * (
                    grid[(i - 1) * n + j]
                    + grid[(i + 1) * n + j]
                    + grid[i * n + (j - 1)]
                    + grid[i * n + (j + 1)]
                )
        # Swap grids
        grid, new_grid = new_grid, grid

    # Sum interior values
    total = 0.0
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            total += grid[i * n + j]

    return total
