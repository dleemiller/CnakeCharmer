"""Scale columns of a Fortran-order matrix by column index factor.

Fills an n x n column-major matrix with hash-derived doubles,
then scales each column j by factor (j + 1) / n, and returns
the sum of all elements.

Keywords: numerical, matrix, scale, fortran order, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(300,))
def fortran_mat_scale(n: int) -> float:
    """Scale columns of hash-filled matrix and return element sum.

    Args:
        n: Matrix dimension (n x n).

    Returns:
        Sum of all scaled matrix elements.
    """
    mask = 0xFFFFFFFF
    matrix = [[0.0] * n for _ in range(n)]
    for j in range(n):
        for i in range(n):
            idx = j * n + i
            h = ((idx * 2654435761) & mask) ^ ((idx * 2246822519) & mask)
            matrix[i][j] = (h & 0xFFFF) / 65535.0

    # Scale each column by (j + 1) / n
    for j in range(n):
        factor = (j + 1) / n
        for i in range(n):
            matrix[i][j] *= factor

    total = 0.0
    for i in range(n):
        for j in range(n):
            total += matrix[i][j]
    return total
