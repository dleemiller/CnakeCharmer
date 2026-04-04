"""Conjugate gradient solver for tridiagonal system.

Keywords: conjugate gradient, linear solver, tridiagonal, optimization, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(5000,))
def conjugate_gradient(n: int) -> float:
    """Solve Ax=b with conjugate gradient on a tridiagonal matrix.

    A has diagonal=4, sub/super-diagonal=-1. b[i]=1.0.
    CG iterations until convergence or max 2*n iterations.
    Returns sum of solution vector.

    Args:
        n: Size of the linear system.

    Returns:
        Sum of the solution vector.
    """
    # Initialize
    x = [0.0] * n

    # r = b - A*x = b (since x=0)
    r = [1.0] * n
    p = [1.0] * n

    rs_old = 0.0
    for i in range(n):
        rs_old += r[i] * r[i]

    max_iter = 2 * n
    ap = [0.0] * n

    for _ in range(max_iter):
        # Compute A*p (tridiagonal: -1, 4, -1)
        for i in range(n):
            ap[i] = 4.0 * p[i]
            if i > 0:
                ap[i] -= p[i - 1]
            if i < n - 1:
                ap[i] -= p[i + 1]

        # alpha = rs_old / (p^T * A*p)
        pap = 0.0
        for i in range(n):
            pap += p[i] * ap[i]

        if abs(pap) < 1e-30:
            break

        alpha = rs_old / pap

        # Update x and r
        rs_new = 0.0
        for i in range(n):
            x[i] += alpha * p[i]
            r[i] -= alpha * ap[i]
            rs_new += r[i] * r[i]

        if rs_new < 1e-20:
            break

        beta = rs_new / rs_old

        for i in range(n):
            p[i] = r[i] + beta * p[i]

        rs_old = rs_new

    total = 0.0
    for i in range(n):
        total += x[i]
    return total
