def jacobi_heat_2d(dim, max_iterations):
    """Solve steady-state heat distribution on a 2D grid using Jacobi iteration.

    Sets up a dim x dim grid with boundary conditions (top=100, left=75,
    right=50, bottom=0) and iteratively applies the Jacobi stencil:
        u_new[i][j] = (u[i+1][j] + u[i-1][j] + u[i][j+1] + u[i][j-1]) / 4
    until max_iterations are exhausted or the solution converges (max change
    across all cells falls below 0.0001).

    Args:
        dim: size of the square grid (must be >= 3)
        max_iterations: maximum number of Jacobi iterations

    Returns:
        The sum of all grid values after convergence or max iterations.
    """
    # Initialize grid with zeros
    u = [[0.0] * dim for _ in range(dim)]

    # Set boundary conditions
    for j in range(dim):
        u[0][j] = 100.0  # top row
    for i in range(dim):
        u[i][0] = 75.0  # left column
        u[i][dim - 1] = 50.0  # right column

    u_new = [row[:] for row in u]

    eps = 0.0001
    for iteration in range(max_iterations):
        # Determine which arrays to read from / write to (ping-pong)
        if iteration % 2 == 0:
            src = u
            dst = u_new
        else:
            src = u_new
            dst = u

        # Apply Jacobi stencil to interior points
        for i in range(1, dim - 1):
            for j in range(1, dim - 1):
                dst[i][j] = (src[i + 1][j] + src[i - 1][j] + src[i][j + 1] + src[i][j - 1]) / 4.0

        # Check convergence every 200 iterations
        if iteration % 200 == 0 and iteration > 0:
            converged = True
            for i in range(dim):
                for j in range(dim):
                    diff = u_new[i][j] - u[i][j]
                    if diff < 0:
                        diff = -diff
                    if diff > eps:
                        converged = False
                        break
                if not converged:
                    break
            if converged:
                break

    # Sum the final grid values (use u_new as the latest result)
    total = 0.0
    result = u_new if (iteration % 2 == 0 or iteration == 0) else u
    for i in range(dim):
        for j in range(dim):
            total += result[i][j]
    return total
