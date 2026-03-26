"""Compute centroids of n deterministic polygons using the signed-area weighted formula.

Keywords: geometry, polygon, centroid, signed area, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def polygon_centroid(n: int) -> tuple:
    """Compute the sum of centroid coordinates over n deterministic 6-sided polygons.

    Each polygon i has 6 vertices with radius r = (j*i + 5) % 40 + 10 at
    angle 2*pi*j/6.  The centroid is computed via the standard signed-area
    weighted formula.

    Args:
        n: Number of polygons.

    Returns:
        Tuple of (total_cx, total_cy, total_area) summed over all polygons.
    """
    total_cx = 0.0
    total_cy = 0.0
    total_area = 0.0
    two_pi_over_6 = 2.0 * math.pi / 6.0

    for i in range(n):
        vx = [0.0] * 6
        vy = [0.0] * 6
        for j in range(6):
            r = (j * i + 5) % 40 + 10
            angle = j * two_pi_over_6
            vx[j] = r * math.cos(angle)
            vy[j] = r * math.sin(angle)

        # Signed area and centroid via shoelace
        area = 0.0
        cx = 0.0
        cy = 0.0
        for j in range(6):
            j1 = (j + 1) % 6
            cross = vx[j] * vy[j1] - vx[j1] * vy[j]
            area += cross
            cx += (vx[j] + vx[j1]) * cross
            cy += (vy[j] + vy[j1]) * cross

        area *= 0.5
        if abs(area) > 1e-12:
            cx /= 6.0 * area
            cy /= 6.0 * area
        else:
            cx = 0.0
            cy = 0.0

        total_cx += cx
        total_cy += cy
        total_area += abs(area)

    return (total_cx, total_cy, total_area)
