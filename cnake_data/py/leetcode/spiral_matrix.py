"""Generate a spiral matrix and compute statistics.

Keywords: leetcode, spiral matrix, generation, traversal, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(1500,))
def spiral_matrix(n: int) -> tuple:
    """Generate an n x n spiral matrix and compute statistics.

    Fills an n x n matrix in spiral order with values 1, 2, 3, ...
    Then computes corner sum, center value, and main diagonal sum.

    Args:
        n: Dimension of the square matrix.

    Returns:
        Tuple of (corner_sum, center_val, diagonal_sum).
    """
    total = n * n
    # Flat array representing n x n matrix
    mat = [0] * total

    top = 0
    bottom = n - 1
    left = 0
    right = n - 1
    val = 1

    while top <= bottom and left <= right:
        # Fill top row
        for col in range(left, right + 1):
            mat[top * n + col] = val
            val += 1
        top += 1

        # Fill right column
        for row in range(top, bottom + 1):
            mat[row * n + right] = val
            val += 1
        right -= 1

        # Fill bottom row
        if top <= bottom:
            for col in range(right, left - 1, -1):
                mat[bottom * n + col] = val
                val += 1
            bottom -= 1

        # Fill left column
        if left <= right:
            for row in range(bottom, top - 1, -1):
                mat[row * n + left] = val
                val += 1
            left += 1

    # Corner sum: mat[0,0] + mat[0,n-1] + mat[n-1,0] + mat[n-1,n-1]
    corner_sum = mat[0] + mat[n - 1] + mat[(n - 1) * n] + mat[(n - 1) * n + n - 1]

    # Center value
    center_val = mat[(n // 2) * n + n // 2]

    # Main diagonal sum
    diagonal_sum = 0
    for i in range(n):
        diagonal_sum += mat[i * n + i]

    return (corner_sum, center_val, diagonal_sum)
