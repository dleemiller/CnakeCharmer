"""QR decomposition via classical Gram-Schmidt.

Keywords: numerical, QR, decomposition, Gram-Schmidt, linear algebra, matrix, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(200,))
def qr_decomposition(n: int) -> tuple:
    """Compute QR decomposition of an n x n matrix via classical Gram-Schmidt.

    Constructs A[i][j] = sin(i * 0.7 + j * 0.3) + 2.0 * (1 if i == j else 0)
    to ensure the matrix is well-conditioned. Decomposes A = QR and returns
    summary statistics from R.

    Args:
        n: Matrix dimension.

    Returns:
        Tuple of (norm of R diagonal, R[0][0], trace of R).
    """
    # Build matrix A (flat, column-major for easier Gram-Schmidt)
    # A_col[j][i] = A[i][j], stored as A_col[j * n + i]
    A = [0.0] * (n * n)
    for i in range(n):
        for j in range(n):
            val = math.sin(i * 0.7 + j * 0.3)
            if i == j:
                val += 2.0
            A[j * n + i] = val  # column-major

    # Q stored column-major: Q[j * n + i] = Q[i][j]
    Q = [0.0] * (n * n)
    # R stored row-major: R[i * n + j]
    R = [0.0] * (n * n)

    # Classical Gram-Schmidt
    for j in range(n):
        # Copy column j of A into v
        v = [0.0] * n
        for i in range(n):
            v[i] = A[j * n + i]

        # Subtract projections onto previous Q columns
        for k in range(j):
            # R[k][j] = dot(Q[:, k], A[:, j])
            dot = 0.0
            for i in range(n):
                dot += Q[k * n + i] * A[j * n + i]
            R[k * n + j] = dot
            for i in range(n):
                v[i] -= dot * Q[k * n + i]

        # R[j][j] = norm(v)
        norm = 0.0
        for i in range(n):
            norm += v[i] * v[i]
        norm = math.sqrt(norm)
        R[j * n + j] = norm

        # Q[:, j] = v / norm
        if norm > 1e-15:
            inv_norm = 1.0 / norm
            for i in range(n):
                Q[j * n + i] = v[i] * inv_norm

    # Compute return values
    # Norm of R diagonal
    diag_norm = 0.0
    for i in range(n):
        d = R[i * n + i]
        diag_norm += d * d
    diag_norm = math.sqrt(diag_norm)

    r00 = R[0]

    # Trace of R
    trace = 0.0
    for i in range(n):
        trace += R[i * n + i]

    return (diag_norm, r00, trace)
