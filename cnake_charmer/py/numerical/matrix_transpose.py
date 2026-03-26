"""Matrix transpose of a flat n x n array.

Keywords: numerical, matrix, transpose, linear algebra, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(800,))
def matrix_transpose(n: int) -> tuple:
    """Transpose an n x n matrix stored as a flat array.

    Constructs A[i][j] = sin(i * 0.01 + j * 0.007) and transposes it in-place
    using the standard swap algorithm. Returns a checksum and corner elements
    for equivalence verification.

    Args:
        n: Matrix dimension.

    Returns:
        Tuple of (checksum, top-right element, bottom-left element).
    """
    # Build matrix (flat)
    A = [0.0] * (n * n)
    for i in range(n):
        for j in range(n):
            A[i * n + j] = math.sin(i * 0.01 + j * 0.007)

    # Transpose in-place
    for i in range(n):
        for j in range(i + 1, n):
            idx_ij = i * n + j
            idx_ji = j * n + i
            tmp = A[idx_ij]
            A[idx_ij] = A[idx_ji]
            A[idx_ji] = tmp

    # Compute checksum: sum of all elements (should be invariant to transpose)
    checksum = 0.0
    for i in range(n * n):
        checksum += A[i]

    # Corner elements after transpose: top-right = A[0][n-1], bottom-left = A[n-1][0]
    top_right = A[0 * n + (n - 1)]
    bottom_left = A[(n - 1) * n + 0]

    return (checksum, top_right, bottom_left)
