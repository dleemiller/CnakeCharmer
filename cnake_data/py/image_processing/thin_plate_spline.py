"""Thin-plate spline (TPS) warp field evaluation.

Evaluates a TPS warp at a grid of output points given a set of control
point correspondences. The TPS kernel is r² log(r²).

Keywords: thin plate spline, TPS, warp, image registration, spatial transform
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(60, 60, 12))
def thin_plate_spline(rows: int, cols: int, n_ctrl: int) -> tuple:
    """Evaluate TPS warp displacement at a grid of output pixels.

    Control points are placed on a regular sub-grid; displacements are
    deterministic sinusoidal offsets.

    Args:
        rows: Output grid height.
        cols: Output grid width.
        n_ctrl: Number of control points (placed on sqrt(n_ctrl)×sqrt(n_ctrl) grid).

    Returns:
        Tuple of (sum_dx, sum_dy, max_disp) — sum and max of displacement magnitudes.
    """
    # Build control points on a regular grid
    n_side = int(math.sqrt(n_ctrl))
    ctrl = []
    weights_x = []
    weights_y = []
    for i in range(n_side):
        for j in range(n_side):
            px = j * (cols - 1) / (n_side - 1) if n_side > 1 else cols / 2
            py = i * (rows - 1) / (n_side - 1) if n_side > 1 else rows / 2
            ctrl.append((px, py))
            # Sinusoidal displacements
            wx = 3.0 * math.sin(i * 0.8 + j * 0.5)
            wy = 2.0 * math.cos(i * 0.6 - j * 0.7)
            weights_x.append(wx)
            weights_y.append(wy)

    m = len(ctrl)

    sum_dx = 0.0
    sum_dy = 0.0
    max_disp = 0.0

    for r in range(rows):
        y = r * (rows - 1) / (rows - 1) if rows > 1 else 0.0
        y = float(r)
        for c in range(cols):
            x = float(c)
            dx = 0.0
            dy = 0.0
            for k in range(m):
                ex = x - ctrl[k][0]
                ey = y - ctrl[k][1]
                r2 = ex * ex + ey * ey
                if r2 > 0.0:
                    kernel = r2 * math.log(r2)
                    dx += weights_x[k] * kernel
                    dy += weights_y[k] * kernel
            sum_dx += dx
            sum_dy += dy
            disp = math.sqrt(dx * dx + dy * dy)
            if disp > max_disp:
                max_disp = disp

    return (sum_dx, sum_dy, max_disp)
