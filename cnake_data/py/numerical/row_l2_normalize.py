"""In-place L2 row normalization of a dense float matrix.

Each row is divided by its L2 norm (Euclidean length). Rows with zero
norm are left unchanged. Returns the sum of all squared elements (which
equals the number of non-zero rows after normalization).

Keywords: normalization, l2, row normalization, linear algebra, numerical, dense matrix
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(300, 200))
def row_l2_normalize(n: int, m: int) -> float:
    """L2-normalize each row of a deterministically generated n×m matrix.

    Args:
        n: Number of rows.
        m: Number of columns.

    Returns:
        Sum of all squared elements after normalization (≈ number of non-zero rows).
    """
    # Generate deterministic matrix values in [-1, 1]
    mat = [[math.sin(i * 1.7 + j * 0.9) for j in range(m)] for i in range(n)]

    # Normalize each row in-place
    for i in range(n):
        norm_sq = 0.0
        for j in range(m):
            norm_sq += mat[i][j] * mat[i][j]
        if norm_sq == 0.0:
            continue
        norm = math.sqrt(norm_sq)
        for j in range(m):
            mat[i][j] /= norm

    # Checksum: sum of squares (should equal count of non-zero rows)
    total = 0.0
    for i in range(n):
        for j in range(m):
            total += mat[i][j] * mat[i][j]
    return total
