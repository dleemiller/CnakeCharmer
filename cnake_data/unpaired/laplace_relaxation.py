def laplace_relaxation(grid_size, num_iters):
    """Solve Laplace's equation on a 2D grid using iterative relaxation.

    Sets the top boundary to 1.0 and all other boundaries to 0.0,
    then iterates the 5-point Laplace stencil for num_iters steps.

    Args:
        grid_size: size of the square grid (must be >= 3)
        num_iters: number of relaxation iterations

    Returns:
        The sum of all grid values after relaxation (a float).
    """
    dx = 0.1
    dy = 0.1
    dx2 = dx * dx
    dy2 = dy * dy

    # Initialize grid with zeros
    u = [[0.0] * grid_size for _ in range(grid_size)]
    # Set top boundary condition
    for j in range(grid_size):
        u[0][j] = 1.0

    for _iteration in range(num_iters):
        for i in range(1, grid_size - 1):
            for j in range(1, grid_size - 1):
                u[i][j] = (
                    (u[i + 1][j] + u[i - 1][j]) * dy2 + (u[i][j + 1] + u[i][j - 1]) * dx2
                ) / (2 * (dx2 + dy2))

    total = 0.0
    for i in range(grid_size):
        for j in range(grid_size):
            total += u[i][j]
    return total
