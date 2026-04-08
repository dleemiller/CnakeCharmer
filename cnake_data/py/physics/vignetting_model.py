"""Radial vignetting model for optical systems.

Models light falloff across an image sensor using the cos^4 law and an
additive polynomial correction term, both common in lens characterization.

Keywords: vignetting, cos4 law, photometric calibration, lens falloff, optical distortion
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(200, 200))
def vignetting_model(rows: int, cols: int) -> tuple:
    """Apply cos^4 + polynomial vignetting to a pixel grid.

    Args:
        rows: Image height.
        cols: Image width.

    Returns:
        Tuple of (total_gain, min_gain, center_gain).
    """
    # Sensor half-diagonal for normalization
    cx = (cols - 1) * 0.5
    cy = (rows - 1) * 0.5
    half_diag = math.sqrt(cx * cx + cy * cy)

    # Polynomial coefficients (common 4th-order model)
    a0 = 1.0
    a2 = -0.35
    a4 = 0.08

    total = 0.0
    min_gain = 2.0
    center_gain = 0.0

    for r in range(rows):
        dy = (r - cy) / half_diag
        for c in range(cols):
            dx = (c - cx) / half_diag
            r2 = dx * dx + dy * dy
            # cos^4 falloff (angle from optical axis)
            cos_theta = 1.0 / math.sqrt(1.0 + r2)
            cos4 = cos_theta * cos_theta * cos_theta * cos_theta
            # Polynomial correction
            poly = a0 + a2 * r2 + a4 * r2 * r2
            gain = cos4 * poly
            total += gain
            if gain < min_gain:
                min_gain = gain
            if r == rows // 2 and c == cols // 2:
                center_gain = gain

    return (total, min_gain, center_gain)
