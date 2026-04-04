"""Gaussian elimination with partial pivoting.

Keywords: numerical, gaussian elimination, linear algebra, solver, pivoting, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(250,))
def gaussian_elimination(n: int) -> tuple:
    """Solve Ax = b via Gaussian elimination with partial pivoting.

    Constructs an n x n matrix A[i][j] = sin(i + j + 1) + 2.0 * (1 if i == j else 0)
    and b[i] = cos(i * 0.3). Solves for x using Gaussian elimination with
    partial pivoting and back substitution.

    Args:
        n: System dimension.

    Returns:
        Tuple of (sum of solution elements, first element, last element).
    """
    # Build augmented matrix [A | b] as flat array of size n * (n+1)
    ncols = n + 1
    M = [0.0] * (n * ncols)
    for i in range(n):
        for j in range(n):
            val = math.sin(i + j + 1)
            if i == j:
                val += 2.0
            M[i * ncols + j] = val
        M[i * ncols + n] = math.cos(i * 0.3)

    # Forward elimination with partial pivoting
    for k in range(n):
        # Find pivot
        max_val = abs(M[k * ncols + k])
        max_row = k
        for i in range(k + 1, n):
            val = abs(M[i * ncols + k])
            if val > max_val:
                max_val = val
                max_row = i

        # Swap rows
        if max_row != k:
            for j in range(ncols):
                tmp = M[k * ncols + j]
                M[k * ncols + j] = M[max_row * ncols + j]
                M[max_row * ncols + j] = tmp

        # Eliminate below
        pivot = M[k * ncols + k]
        if abs(pivot) < 1e-15:
            continue
        for i in range(k + 1, n):
            factor = M[i * ncols + k] / pivot
            for j in range(k, ncols):
                M[i * ncols + j] -= factor * M[k * ncols + j]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i * ncols + n]
        for j in range(i + 1, n):
            s -= M[i * ncols + j] * x[j]
        if abs(M[i * ncols + i]) > 1e-15:
            x[i] = s / M[i * ncols + i]

    # Compute return values
    x_sum = 0.0
    for i in range(n):
        x_sum += x[i]

    return (x_sum, x[0], x[n - 1])
