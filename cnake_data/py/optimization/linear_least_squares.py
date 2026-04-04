"""Linear least squares via normal equations.

Keywords: linear least squares, normal equations, polynomial fit, optimization, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def linear_least_squares(n: int) -> float:
    """Solve overdetermined Ax=b in least squares sense.

    A is n x 3 with A[i] = [1, i*0.01, (i*0.01)^2]. b[i] = sin(i*0.01).
    Uses normal equations A^T*A*x = A^T*b (3x3 system).
    Returns sum of coefficients.

    Args:
        n: Number of data points (rows of A).

    Returns:
        Sum of the three fitted coefficients.
    """
    # Build A^T*A (3x3) and A^T*b (3x1)
    ata = [[0.0] * 3 for _ in range(3)]
    atb = [0.0] * 3

    for i in range(n):
        t = i * 0.01
        t2 = t * t
        bi = math.sin(t)

        # A[i] = [1, t, t^2]
        ata[0][0] += 1.0
        ata[0][1] += t
        ata[0][2] += t2
        ata[1][1] += t2
        ata[1][2] += t * t2
        ata[2][2] += t2 * t2

        atb[0] += bi
        atb[1] += t * bi
        atb[2] += t2 * bi

    # Symmetric
    ata[1][0] = ata[0][1]
    ata[2][0] = ata[0][2]
    ata[2][1] = ata[1][2]

    # Solve 3x3 via Cramer's rule
    a = ata
    det = (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )

    if abs(det) < 1e-30:
        return 0.0

    inv_det = 1.0 / det

    x0 = inv_det * (
        atb[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - atb[1] * (a[0][1] * a[2][2] - a[0][2] * a[2][1])
        + atb[2] * (a[0][1] * a[1][2] - a[0][2] * a[1][1])
    )
    x1 = inv_det * (
        a[0][0] * (atb[1] * a[2][2] - atb[2] * a[2][1])
        - atb[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (atb[2] * a[1][0] - atb[1] * a[2][0])
    )
    x2 = inv_det * (
        a[0][0] * (a[1][1] * atb[2] - atb[1] * a[2][1])
        - a[0][1] * (a[1][0] * atb[2] - atb[1] * a[2][0])
        + atb[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )

    return x0 + x1 + x2
