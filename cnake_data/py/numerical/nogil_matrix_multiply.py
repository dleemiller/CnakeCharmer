"""Matrix multiply with GIL release, returning trace.

Keywords: matrix, multiply, nogil, numerical, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(200,))
def nogil_matrix_multiply(n: int) -> float:
    """Multiply two n x n matrices and return the trace.

    A[i][j] = (i * 3 + j * 7 + 1) % 100 / 10.0
    B[i][j] = (i * 11 + j * 5 + 3) % 100 / 10.0

    Args:
        n: Matrix dimension.

    Returns:
        Trace of the product matrix C = A * B.
    """
    A = [[0.0] * n for _ in range(n)]
    B = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            A[i][j] = ((i * 3 + j * 7 + 1) % 100) / 10.0
            B[i][j] = ((i * 11 + j * 5 + 3) % 100) / 10.0

    trace = 0.0
    for i in range(n):
        s = 0.0
        for k in range(n):
            s += A[i][k] * B[k][i]
        trace += s

    return trace
