"""Levenberg-Marquardt fitting of sinusoidal model.

Keywords: levenberg-marquardt, curve fitting, sinusoidal, optimization, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def levenberg_marquardt(n: int) -> float:
    """Fit y = a*sin(b*x + c) to n data points using simplified LM.

    x[i] = i*0.01, y[i] = 3*sin(2*x[i] + 1) + noise.
    5 LM iterations. Returns sum of squared residuals.

    Args:
        n: Number of data points.

    Returns:
        Sum of squared residuals after fitting.
    """
    # Generate data
    x_data = [0.0] * n
    y_data = [0.0] * n
    for i in range(n):
        x_data[i] = i * 0.01
        noise = 0.05 * ((i * 13 + 7) % 100 - 50) / 50.0
        y_data[i] = 3.0 * math.sin(2.0 * x_data[i] + 1.0) + noise

    # Initial guess
    a = 2.0
    b = 1.5
    c = 0.5
    lam = 0.01  # damping parameter

    for _ in range(5):
        # Build 3x3 normal equations: (J^T J + lambda*I) * dp = J^T r
        jtj = [[0.0] * 3 for _ in range(3)]
        jtr = [0.0] * 3

        for i in range(n):
            xi = x_data[i]
            yi = y_data[i]
            arg = b * xi + c
            sin_val = math.sin(arg)
            cos_val = math.cos(arg)
            pred = a * sin_val
            ri = yi - pred

            # Jacobian row: [sin(bx+c), a*x*cos(bx+c), a*cos(bx+c)]
            j0 = sin_val
            j1 = a * xi * cos_val
            j2 = a * cos_val

            jtr[0] += j0 * ri
            jtr[1] += j1 * ri
            jtr[2] += j2 * ri

            jtj[0][0] += j0 * j0
            jtj[0][1] += j0 * j1
            jtj[0][2] += j0 * j2
            jtj[1][1] += j1 * j1
            jtj[1][2] += j1 * j2
            jtj[2][2] += j2 * j2

        # Symmetric
        jtj[1][0] = jtj[0][1]
        jtj[2][0] = jtj[0][2]
        jtj[2][1] = jtj[1][2]

        # Add damping
        for k in range(3):
            jtj[k][k] += lam

        # Solve 3x3 system via Cramer's rule
        m = jtj
        det = (
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        )

        if abs(det) < 1e-30:
            break

        inv_det = 1.0 / det
        dp0 = inv_det * (
            jtr[0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - jtr[1] * (m[0][1] * m[2][2] - m[0][2] * m[2][1])
            + jtr[2] * (m[0][1] * m[1][2] - m[0][2] * m[1][1])
        )
        dp1 = inv_det * (
            m[0][0] * (jtr[1] * m[2][2] - jtr[2] * m[2][1])
            - jtr[0] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (jtr[2] * m[1][0] - jtr[1] * m[2][0])
        )  # Fixed sign
        dp2 = inv_det * (
            m[0][0] * (m[1][1] * jtr[2] - jtr[1] * m[2][1])
            - m[0][1] * (m[1][0] * jtr[2] - jtr[1] * m[2][0])
            + jtr[0] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        )

        a += dp0
        b += dp1
        c += dp2

    # Compute final residual sum of squares
    rss = 0.0
    for i in range(n):
        pred = a * math.sin(b * x_data[i] + c)
        diff = y_data[i] - pred
        rss += diff * diff

    return rss
