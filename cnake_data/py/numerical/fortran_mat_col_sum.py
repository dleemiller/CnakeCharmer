"""Column sums of a Fortran-order (column-major) matrix.

Fills an n x n matrix in column-major order with hash-derived
doubles, then sums each column and returns the total of all
column sums.

Keywords: numerical, matrix, column sum, fortran order, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(300,))
def fortran_mat_col_sum(n: int) -> float:
    """Compute sum of all column sums of a hash-filled n x n matrix.

    Matrix is logically stored in column-major order.

    Args:
        n: Matrix dimension (n x n).

    Returns:
        Total of all column sums.
    """
    mask = 0xFFFFFFFF
    # Build matrix in column-major logical order
    matrix = [[0.0] * n for _ in range(n)]
    for j in range(n):
        for i in range(n):
            idx = j * n + i  # column-major index
            h = ((idx * 2654435761) & mask) ^ ((idx * 2246822519) & mask)
            matrix[i][j] = (h & 0xFFFF) / 65535.0

    total = 0.0
    for j in range(n):
        col_sum = 0.0
        for i in range(n):
            col_sum += matrix[i][j]
        total += col_sum
    return total
