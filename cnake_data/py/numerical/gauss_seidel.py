"""Gauss-Seidel iterative solver for a diagonally dominant system.

Keywords: numerical, linear algebra, iterative solver, gauss-seidel, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(300,))
def gauss_seidel(n: int) -> tuple:
    """Solve an n x n diagonally dominant linear system using Gauss-Seidel.

    Constructs A where A[i][i] = 2*n, A[i][j] = ((i*7+j*3) % 5) / n,
    and b[i] = sum(A[i][j] for j in range(n)).
    Solution should converge to all ones.

    Args:
        n: System size.

    Returns:
        Tuple of (solution_sum, x_first, x_last).
    """
    # Build diagonally dominant matrix
    A = [[0.0] * n for _ in range(n)]
    b = [0.0] * n

    for i in range(n):
        row_sum = 0.0
        for j in range(n):
            if i != j:
                val = ((i * 7 + j * 3) % 5) / float(n)
                A[i][j] = val
                row_sum += val
        A[i][i] = 2.0 * n
        # b = A * [1,1,...,1] so true solution is [1,1,...,1]
        b[i] = row_sum + 2.0 * n

    # Gauss-Seidel iteration
    x = [0.0] * n
    max_iter = 200

    for _iteration in range(max_iter):
        for i in range(n):
            sigma = 0.0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x[j]
            x[i] = (b[i] - sigma) / A[i][i]

    solution_sum = 0.0
    for i in range(n):
        solution_sum += x[i]

    return (solution_sum, x[0], x[n - 1])
