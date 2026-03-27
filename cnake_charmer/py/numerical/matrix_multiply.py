"""Matrix multiplication returning trace, corner sum, and mid element.

Keywords: matrix, multiply, linear algebra, numerical, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(150,))
def matrix_multiply(n: int) -> tuple:
    """Multiply two n x n deterministic matrices and return summary values.

    Creates A[i][j] = i + j, B[i][j] = i - j, computes C = A * B.

    Args:
        n: Matrix dimension.

    Returns:
        Tuple of (trace, corner_sum, mid_element).
        corner_sum = C[0][0] + C[0][n-1] + C[n-1][0] + C[n-1][n-1].
        mid_element = C[n//2][n//2].
    """
    A = [[0.0] * n for _ in range(n)]
    B = [[0.0] * n for _ in range(n)]
    C = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            A[i][j] = float(i + j)
            B[i][j] = float(i - j)

    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s

    trace = 0.0
    for i in range(n):
        trace += C[i][i]

    corner_sum = C[0][0] + C[0][n - 1] + C[n - 1][0] + C[n - 1][n - 1]
    mid = n // 2
    mid_element = C[mid][mid]

    return (trace, corner_sum, mid_element)
