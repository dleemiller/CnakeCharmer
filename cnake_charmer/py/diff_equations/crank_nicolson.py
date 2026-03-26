"""Solve 1D heat equation using Crank-Nicolson scheme.

n cells, 1000 time steps. u(x,0) = sin(pi*x/n)*100. Returns sum of final values.

Keywords: PDE, heat equation, Crank-Nicolson, tridiagonal, Thomas algorithm, numerical
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000,))
def crank_nicolson(n: int) -> float:
    """Solve 1D heat equation using Crank-Nicolson scheme.

    Args:
        n: Number of spatial cells.

    Returns:
        Sum of final temperature values.
    """
    num_steps = 1000
    dx = 1.0 / n
    dt = 0.5 * dx * dx  # stable time step
    r = dt / (dx * dx)

    # Initial condition: u(x,0) = sin(pi*x/n)*100
    u = [0.0] * n
    for i in range(n):
        u[i] = math.sin(math.pi * i / n) * 100.0

    # Tridiagonal coefficients for Crank-Nicolson
    # (I - 0.5*r*A) u^{n+1} = (I + 0.5*r*A) u^n
    alpha = -0.5 * r
    beta = 1.0 + r
    gamma = -0.5 * r

    alpha_rhs = 0.5 * r
    beta_rhs = 1.0 - r
    gamma_rhs = 0.5 * r

    # Temporary arrays for Thomas algorithm
    cp = [0.0] * n
    dp = [0.0] * n
    rhs = [0.0] * n

    for _ in range(num_steps):
        # Build RHS
        rhs[0] = beta_rhs * u[0] + gamma_rhs * u[1]
        for i in range(1, n - 1):
            rhs[i] = alpha_rhs * u[i - 1] + beta_rhs * u[i] + gamma_rhs * u[i + 1]
        rhs[n - 1] = alpha_rhs * u[n - 2] + beta_rhs * u[n - 1]

        # Thomas algorithm - forward sweep
        cp[0] = gamma / beta
        dp[0] = rhs[0] / beta
        for i in range(1, n):
            cp[i] = gamma / (beta - alpha * cp[i - 1])
            dp[i] = (rhs[i] - alpha * dp[i - 1]) / (beta - alpha * cp[i - 1])

        # Back substitution
        u[n - 1] = dp[n - 1]
        for i in range(n - 2, -1, -1):
            u[i] = dp[i] - cp[i] * u[i + 1]

    total = 0.0
    for i in range(n):
        total += u[i]

    return total
