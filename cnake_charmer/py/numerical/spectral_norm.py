"""Spectral norm of a matrix via power iteration.

Keywords: spectral norm, eigenvalue, power iteration, linear algebra, matrix
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(300,))
def spectral_norm(n):
    """Compute spectral norm of the n×n truncation of matrix A.

    A[i,j] = 1.0 / ((i+j)*(i+j+1)/2 + i + 1)

    Uses 10 iterations of the power method on B = A^T * A.

    Args:
        n: Matrix dimension.

    Returns:
        Tuple of (norm_value, u_checksum, v_checksum).
    """
    u = [1.0] * n
    v = [1.0] * n
    tmp = [0.0] * n

    for _ in range(10):
        # v = A^T * A * u
        # tmp = A * u
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += 1.0 / ((i + j) * (i + j + 1) // 2 + i + 1) * u[j]
            tmp[i] = s
        # v = A^T * tmp
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += 1.0 / ((j + i) * (j + i + 1) // 2 + j + 1) * tmp[j]
            v[i] = s

        # u = A^T * A * v
        # tmp = A * v
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += 1.0 / ((i + j) * (i + j + 1) // 2 + i + 1) * v[j]
            tmp[i] = s
        # u = A^T * tmp
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += 1.0 / ((j + i) * (j + i + 1) // 2 + j + 1) * tmp[j]
            u[i] = s

    vbv = 0.0
    vv = 0.0
    for i in range(n):
        vbv += u[i] * v[i]
        vv += v[i] * v[i]

    norm_val = math.sqrt(vbv / vv)

    u_check = 0.0
    v_check = 0.0
    k = min(10, n)
    for i in range(k):
        u_check += u[i]
        v_check += v[i]

    return (norm_val, u_check, v_check)
