"""
Element-wise addition of two n*n matrices, returning the sum of all elements in the result.

Keywords: numerical, matrix, addition, element-wise, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(300,))
def memview_mat_add(n: int) -> float:
    """Add two n*n matrices element-wise and return the total sum.

    Matrix A: A[i][j] = ((i * 53 + j * 37 + 7) % 500) / 5.0
    Matrix B: B[i][j] = ((i * 41 + j * 67 + 13) % 500) / 5.0

    Args:
        n: Dimension of the square matrices.

    Returns:
        Sum of all elements in A + B.
    """
    A = [0.0] * (n * n)
    B = [0.0] * (n * n)
    for i in range(n):
        for j in range(n):
            A[i * n + j] = ((i * 53 + j * 37 + 7) % 500) / 5.0
            B[i * n + j] = ((i * 41 + j * 67 + 13) % 500) / 5.0

    total = 0.0
    for i in range(n):
        for j in range(n):
            total += A[i * n + j] + B[i * n + j]

    return total
