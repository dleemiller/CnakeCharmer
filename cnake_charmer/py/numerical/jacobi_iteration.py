"""Solve tridiagonal system with Jacobi iteration.

Keywords: jacobi, iteration, linear algebra, tridiagonal, solver, numerical, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(5000,))
def jacobi_iteration(n: int) -> float:
    """Solve Ax=b with Jacobi iteration on a tridiagonal matrix.

    A has diagonal=-2, sub/super-diagonal=1. rhs[i]=1.0.
    Performs 1000 iterations on n unknowns. Returns sum of solution.

    Args:
        n: Number of unknowns.

    Returns:
        Sum of the solution vector after 1000 iterations.
    """
    iterations = 1000

    x = [0.0] * n
    x_new = [0.0] * n
    rhs = [1.0] * n

    for _ in range(iterations):
        for i in range(n):
            sigma = 0.0
            if i > 0:
                sigma += x[i - 1]
            if i < n - 1:
                sigma += x[i + 1]
            x_new[i] = (rhs[i] - sigma) / (-2.0)
        x, x_new = x_new, x

    total = 0.0
    for i in range(n):
        total += x[i]
    return total
