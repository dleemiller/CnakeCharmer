"""
Transpose an n*n matrix and compute the trace of A + A^T using flat lists.

Keywords: numerical, matrix, transpose, trace, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(300,))
def memview_mat_transpose(n: int) -> float:
    """Transpose n*n matrix, return trace of A + A^T.

    Matrix is generated deterministically: A[i][j] = ((i * 97 + j * 31 + 17) % 1000) / 10.0.

    Args:
        n: Dimension of the square matrix.

    Returns:
        Trace of A + A^T as a float.
    """
    # Build flat matrix A
    A = [0.0] * (n * n)
    for i in range(n):
        for j in range(n):
            A[i * n + j] = ((i * 97 + j * 31 + 17) % 1000) / 10.0

    # Build transpose AT
    AT = [0.0] * (n * n)
    for i in range(n):
        for j in range(n):
            AT[i * n + j] = A[j * n + i]

    # Compute trace of A + AT
    trace = 0.0
    for i in range(n):
        trace += A[i * n + i] + AT[i * n + i]

    return trace
