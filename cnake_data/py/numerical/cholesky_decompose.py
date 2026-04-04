"""Cholesky decomposition.

Keywords: numerical, cholesky, decomposition, linear algebra, matrix, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(300,))
def cholesky_decompose(n: int) -> float:
    """Compute Cholesky decomposition and return sum of diagonal of L.

    Constructs an n x n positive-definite matrix A[i][j] = 1 / (abs(i-j) + 1),
    then computes the lower triangular matrix L such that A = L * L^T.

    Args:
        n: Matrix dimension.

    Returns:
        Sum of the diagonal elements of L.
    """
    # Build positive-definite matrix (flat)
    A = [0.0] * (n * n)
    for i in range(n):
        for j in range(n):
            A[i * n + j] = 1.0 / (abs(i - j) + 1)

    # Cholesky decomposition: L stored in flat array
    L = [0.0] * (n * n)

    for i in range(n):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s += L[i * n + k] * L[j * n + k]

            if i == j:
                L[i * n + j] = math.sqrt(A[i * n + i] - s)
            else:
                L[i * n + j] = (A[i * n + j] - s) / L[j * n + j]

    # Sum diagonal of L
    diag_sum = 0.0
    for i in range(n):
        diag_sum += L[i * n + i]

    return diag_sum
