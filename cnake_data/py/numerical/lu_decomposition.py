"""LU decomposition.

Keywords: numerical, lu decomposition, linear algebra, matrix, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(300,))
def lu_decomposition(n: int) -> float:
    """Compute LU decomposition and return sum of diagonal of U.

    Constructs an n x n matrix A[i][j] = (i * j + i + j + 1) % 100,
    then computes the LU decomposition without pivoting. Returns the
    sum of the diagonal elements of U.

    Args:
        n: Matrix dimension.

    Returns:
        Sum of the diagonal elements of U.
    """
    # Build matrix (flat)
    A = [0.0] * (n * n)
    for i in range(n):
        for j in range(n):
            A[i * n + j] = float((i * j + i + j + 1) % 100)

    # LU decomposition in-place (Doolittle algorithm)
    # L has 1s on diagonal, U is upper triangular
    # After decomposition, A contains both L and U
    for k in range(n):
        for i in range(k + 1, n):
            if A[k * n + k] == 0.0:
                continue
            A[i * n + k] = A[i * n + k] / A[k * n + k]
            for j in range(k + 1, n):
                A[i * n + j] -= A[i * n + k] * A[k * n + j]

    # Sum diagonal of U (stored on diagonal of A after decomposition)
    diag_sum = 0.0
    for i in range(n):
        diag_sum += A[i * n + i]

    return diag_sum
