"""Solve a tridiagonal system using the Thomas algorithm.

Keywords: tridiagonal, linear system, Thomas algorithm, numerical, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def tridiagonal_solve(n: int) -> float:
    """Solve tridiagonal system Ax=b of size n using Thomas algorithm.

    Diagonals: a[i]=1 (sub), b[i]=4 (main), c[i]=1 (super).
    Right-hand side: rhs[i] = sin(i * 0.1).

    Args:
        n: System size.

    Returns:
        Sum of the solution vector as a float.
    """
    # Forward sweep
    cp = [0.0] * n
    dp = [0.0] * n

    # First row
    cp[0] = 1.0 / 4.0
    dp[0] = math.sin(0.0) / 4.0

    for i in range(1, n):
        rhs_i = math.sin(i * 0.1)
        denom = 4.0 - 1.0 * cp[i - 1]
        cp[i] = 1.0 / denom if i < n - 1 else 0.0
        dp[i] = (rhs_i - 1.0 * dp[i - 1]) / denom

    # Back substitution
    result = dp[n - 1]
    total = result
    for i in range(n - 2, -1, -1):
        result = dp[i] - cp[i] * result
        total += result

    return total
