"""Fit exponential model using Gauss-Newton least squares.

Keywords: least squares, gauss-newton, exponential fit, optimization, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(10000,))
def least_squares(n: int) -> float:
    """Fit y = a*exp(b*x) to n data points using Gauss-Newton iteration.

    x[i] = i*0.01, y[i] = 2.5*exp(0.3*x[i]) + noise.
    Performs 10 Gauss-Newton iterations. Returns a + b.

    Args:
        n: Number of data points.

    Returns:
        Sum of fitted parameters a + b.
    """
    # Generate data
    x_data = [0.0] * n
    y_data = [0.0] * n
    for i in range(n):
        x_data[i] = i * 0.01
        y_data[i] = 2.5 * math.exp(0.3 * x_data[i]) + 0.1 * ((i * 7 + 3) % 100 - 50) / 50.0

    # Initial guess
    a = 1.0
    b = 0.1

    # Gauss-Newton: 10 iterations
    for _ in range(10):
        # Build J^T J and J^T r (2x2 normal equations)
        jtj00 = 0.0
        jtj01 = 0.0
        jtj11 = 0.0
        jtr0 = 0.0
        jtr1 = 0.0

        for i in range(n):
            xi = x_data[i]
            yi = y_data[i]
            bxi = b * xi
            if bxi > 500.0:
                bxi = 500.0
            elif bxi < -500.0:
                bxi = -500.0
            eb = math.exp(bxi)
            pred = a * eb
            ri = yi - pred
            # Jacobian: d/da = exp(b*x), d/db = a*x*exp(b*x)
            j0 = eb
            j1 = a * xi * eb

            jtj00 += j0 * j0
            jtj01 += j0 * j1
            jtj11 += j1 * j1
            jtr0 += j0 * ri
            jtr1 += j1 * ri

        # Solve 2x2 system: [jtj00, jtj01; jtj01, jtj11] * [da, db] = [jtr0, jtr1]
        det = jtj00 * jtj11 - jtj01 * jtj01
        if abs(det) < 1e-30:
            break
        da = (jtj11 * jtr0 - jtj01 * jtr1) / det
        db = (jtj00 * jtr1 - jtj01 * jtr0) / det

        a += da
        b += db

    return a + b
