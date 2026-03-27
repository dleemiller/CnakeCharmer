"""Compute 3x3 matrix determinants using flat 9-element array.

Keywords: numerical, matrix, determinant, 3x3, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def stack_matrix_det(n: int) -> float:
    """Compute determinants of n 3x3 matrices, return sum.

    Args:
        n: Number of matrices.

    Returns:
        Sum of all determinants.
    """
    total = 0.0

    for k in range(n):
        mat = [0.0] * 9
        seed = k * 2654435761 + 17
        for i in range(9):
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            mat[i] = (seed % 10000) / 1000.0 - 5.0

        # det of 3x3:
        # mat[0]*(mat[4]*mat[8]-mat[5]*mat[7])
        # - mat[1]*(mat[3]*mat[8]-mat[5]*mat[6])
        # + mat[2]*(mat[3]*mat[7]-mat[4]*mat[6])
        det = (
            mat[0] * (mat[4] * mat[8] - mat[5] * mat[7])
            - mat[1] * (mat[3] * mat[8] - mat[5] * mat[6])
            + mat[2] * (mat[3] * mat[7] - mat[4] * mat[6])
        )
        total += det

    return total
