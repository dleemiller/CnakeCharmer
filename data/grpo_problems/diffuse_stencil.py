def diffuse_stencil(grid_size, num_steps):
    """Apply 2D heat diffusion using a forward-Euler stencil.

    Creates a grid_size x grid_size grid with an initial heat source at the center,
    then evolves it for num_steps using a 5-point stencil with diffusion coefficient mu=0.1.

    Args:
        grid_size: size of the square grid (must be >= 4)
        num_steps: number of diffusion time steps to apply

    Returns:
        The sum of all grid values after diffusion (a float).
    """
    # Initialize grid with zeros
    u = [[0.0] * grid_size for _ in range(grid_size)]
    temp = [[0.0] * grid_size for _ in range(grid_size)]

    # Place initial heat source at center
    mid = grid_size // 2
    u[mid][mid] = 1000.0

    mu = 0.1

    for _n in range(num_steps):
        # Apply 5-point stencil
        for i in range(1, grid_size - 1):
            for j in range(1, grid_size - 1):
                temp[i][j] = u[i][j] + mu * (
                    u[i + 1][j]
                    - 2 * u[i][j]
                    + u[i - 1][j]
                    + u[i][j + 1]
                    - 2 * u[i][j]
                    + u[i][j - 1]
                )
        # Copy temp back to u
        for i in range(grid_size):
            for j in range(grid_size):
                u[i][j] = temp[i][j]
                temp[i][j] = 0.0

    # Return sum of all grid values
    total = 0.0
    for i in range(grid_size):
        for j in range(grid_size):
            total += u[i][j]
    return total
