"""
Matrix multiplication (naive O(n^3) implementation).

Keywords: matrix, multiply, linear algebra, numerical, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(50,))
def matrix_multiply(n: int) -> list[list[float]]:
    """Multiply two n×n identity-like matrices and return the result.

    Creates two matrices where A[i][j] = i + j and B[i][j] = i - j,
    then computes C = A × B using naive triple-loop multiplication.

    Args:
        n: Matrix dimension.

    Returns:
        The n×n result matrix as a list of lists.
    """
    A = [[float(i + j) for j in range(n)] for i in range(n)]
    B = [[float(i - j) for j in range(n)] for i in range(n)]
    C = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s

    return C
