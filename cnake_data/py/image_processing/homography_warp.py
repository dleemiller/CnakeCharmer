"""Projective homography warp on a pixel grid.

Applies a 3x3 homography matrix to every pixel in an n×n image,
computing the warped destination coordinates and accumulating statistics.

Keywords: homography, projective transform, warp, geometric vision, calibration
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(120, 120))
def homography_warp(rows: int, cols: int) -> tuple:
    """Apply a fixed 3x3 homography to a pixel grid and return warp statistics.

    Args:
        rows: Image height.
        cols: Image width.

    Returns:
        Tuple of (sum_dx, sum_dy, count_inbounds) where sum_dx/sum_dy are sums
        of warped x/y coordinates for in-bounds pixels.
    """
    # Deterministic homography: slight perspective + rotation + scale
    h00 = 1.05
    h01 = 0.02
    h02 = 5.0
    h10 = -0.01
    h11 = 0.98
    h12 = 3.0
    h20 = 0.0001
    h21 = 0.00005
    h22 = 1.0

    sum_dx = 0.0
    sum_dy = 0.0
    count = 0

    for r in range(rows):
        for c in range(cols):
            # Apply H * [c, r, 1]^T
            w = h20 * c + h21 * r + h22
            dx = (h00 * c + h01 * r + h02) / w
            dy = (h10 * c + h11 * r + h12) / w
            if 0.0 <= dx < cols and 0.0 <= dy < rows:
                sum_dx += dx
                sum_dy += dy
                count += 1

    return (sum_dx, sum_dy, count)
