"""Brown-Conrady radial and tangential lens distortion model.

Applies the standard camera lens distortion model to a grid of normalized
image coordinates, computing distorted positions and their statistics.

Keywords: lens distortion, radial distortion, tangential distortion, Brown-Conrady, camera calibration
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(200, 200))
def radial_distortion(rows: int, cols: int) -> tuple:
    """Apply Brown-Conrady distortion to a normalized coordinate grid.

    Normalized coordinates span [-1, 1] in x and [-1, 1] in y.

    Args:
        rows: Grid height.
        cols: Grid width.

    Returns:
        Tuple of (sum_xd, sum_yd, max_r2) — sum of distorted x/y coords
        and the maximum r² encountered.
    """
    # Fixed distortion coefficients
    k1 = -0.28
    k2 = 0.08
    p1 = 0.001
    p2 = -0.0015

    sum_xd = 0.0
    sum_yd = 0.0
    max_r2 = 0.0

    for r in range(rows):
        y = -1.0 + 2.0 * r / (rows - 1)
        for c in range(cols):
            x = -1.0 + 2.0 * c / (cols - 1)
            r2 = x * x + y * y
            radial = 1.0 + k1 * r2 + k2 * r2 * r2
            xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            sum_xd += xd
            sum_yd += yd
            if r2 > max_r2:
                max_r2 = r2

    return (sum_xd, sum_yd, max_r2)
