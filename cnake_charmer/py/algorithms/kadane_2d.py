"""Maximum sum submatrix using 2D Kadane's algorithm.

Keywords: algorithms, kadane, 2d, max submatrix, dynamic programming, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(200,))
def kadane_2d(n: int) -> int:
    """Find the maximum sum submatrix in an n x n matrix.

    Matrix M[i][j] = ((i*7 + j*13) % 201) - 100.
    Uses O(n^3) algorithm: fix left/right columns, Kadane on column sums.

    Args:
        n: Matrix dimension.

    Returns:
        Maximum submatrix sum.
    """
    # Build matrix
    matrix = [[(((i * 7 + j * 13) % 201) - 100) for j in range(n)] for i in range(n)]

    best = matrix[0][0]

    for left in range(n):
        col_sum = [0] * n
        for right in range(left, n):
            for row in range(n):
                col_sum[row] += matrix[row][right]

            # Kadane on col_sum
            current = col_sum[0]
            max_here = col_sum[0]
            for row in range(1, n):
                if current + col_sum[row] > col_sum[row]:
                    current = current + col_sum[row]
                else:
                    current = col_sum[row]
                if current > max_here:
                    max_here = current
            if max_here > best:
                best = max_here

    return best
