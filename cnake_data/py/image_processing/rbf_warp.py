"""Gaussian radial basis function (RBF) image warp.

Computes displacement at every grid point as a weighted sum of Gaussian
RBF kernels centered at control points — an alternative kernel to TPS.

Keywords: RBF, radial basis function, Gaussian kernel, warp, image registration
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(60, 60, 12, 20.0))
def rbf_warp(rows: int, cols: int, n_ctrl: int, sigma: float) -> tuple:
    """Evaluate Gaussian RBF warp displacement at a grid of pixels.

    Control points are placed on a regular sub-grid with sinusoidal weights.
    Kernel: k(r) = exp(-r² / σ²).

    Args:
        rows: Output grid height.
        cols: Output grid width.
        n_ctrl: Number of control points (sqrt(n_ctrl)×sqrt(n_ctrl) grid).
        sigma: Gaussian kernel width.

    Returns:
        Tuple of (sum_dx, sum_dy, max_disp).
    """
    n_side = int(math.sqrt(n_ctrl))
    m = n_side * n_side
    ctrl_x = []
    ctrl_y = []
    weights_x = []
    weights_y = []
    for i in range(n_side):
        for j in range(n_side):
            px = j * (cols - 1) / (n_side - 1) if n_side > 1 else cols / 2.0
            py = i * (rows - 1) / (n_side - 1) if n_side > 1 else rows / 2.0
            ctrl_x.append(px)
            ctrl_y.append(py)
            weights_x.append(3.0 * math.sin(i * 0.8 + j * 0.5))
            weights_y.append(2.0 * math.cos(i * 0.6 - j * 0.7))

    inv_s2 = 1.0 / (sigma * sigma)
    sum_dx = 0.0
    sum_dy = 0.0
    max_disp = 0.0

    for r in range(rows):
        y = float(r)
        for c in range(cols):
            x = float(c)
            dx = 0.0
            dy = 0.0
            for k in range(m):
                ex = x - ctrl_x[k]
                ey = y - ctrl_y[k]
                r2 = ex * ex + ey * ey
                kernel = math.exp(-r2 * inv_s2)
                dx += weights_x[k] * kernel
                dy += weights_y[k] * kernel
            sum_dx += dx
            sum_dy += dy
            disp = math.sqrt(dx * dx + dy * dy)
            if disp > max_disp:
                max_disp = disp

    return (sum_dx, sum_dy, max_disp)
