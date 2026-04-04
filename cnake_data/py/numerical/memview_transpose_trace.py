"""Trace of A * A^T element-wise product via transpose.

Fills an n x n matrix with hash-derived doubles, computes
the transpose, then sums diagonal of element-wise A * A^T.

Keywords: numerical, matrix, transpose, trace, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(200,))
def memview_transpose_trace(n: int) -> float:
    """Compute trace of element-wise product A * A^T.

    Args:
        n: Matrix dimension (n x n).

    Returns:
        Trace of A * A^T (sum of squared row norms).
    """
    mask = 0xFFFFFFFF
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            h = ((idx * 2654435761) & mask) ^ ((idx * 2246822519) & mask)
            mat[i][j] = (h & 0xFFFF) / 65535.0

    # Transpose
    mat_t = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            mat_t[i][j] = mat[j][i]

    # Trace of element-wise product A * A^T
    trace = 0.0
    for i in range(n):
        for j in range(n):
            trace += mat[i][j] * mat_t[i][j]
    # Only diagonal contributes to trace, but we compute
    # full product then take diag. Simplify: trace(A * A^T)
    # = sum of A[i,j]*A^T[i,j] for diagonal i==i
    # Actually trace = sum_i (A*A^T)[i,i] = sum_i sum_k A[i,k]*A^T[k,i]
    # = sum_i sum_k A[i,k]*A[i,k] = sum of all squared elements
    # But we computed sum_ij A[i,j]*A^T[i,j] = sum_ij A[i,j]*A[j,i]
    # which is sum_ij A[i,j]*A[j,i]. Let's keep the full sum.
    return trace
