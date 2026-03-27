"""Compute polygon area and centroid using the shoelace formula.

Keywords: polygon, area, centroid, shoelace, geometry, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def polygon_area_centroid(n: int) -> tuple:
    """Generate an n-vertex polygon and compute its signed area and centroid.

    Vertices are placed on a distorted circle to form a non-self-intersecting
    polygon. Uses the shoelace formula for area and centroid computation.

    Args:
        n: Number of polygon vertices.

    Returns:
        Tuple of (area, centroid_x, centroid_y).
    """
    import math

    # Generate polygon vertices on a distorted circle
    xs = [0.0] * n
    ys = [0.0] * n
    for i in range(n):
        angle = 2.0 * math.pi * i / n
        # Deterministic radius perturbation
        h = ((i * 2654435761) >> 8) & 0xFFFF
        r = 10.0 + (h % 500) / 100.0
        xs[i] = r * math.cos(angle)
        ys[i] = r * math.sin(angle)

    # Shoelace formula for signed area
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j] - xs[j] * ys[i]
    area *= 0.5

    # Centroid
    cx = 0.0
    cy = 0.0
    for i in range(n):
        j = (i + 1) % n
        cross = xs[i] * ys[j] - xs[j] * ys[i]
        cx += (xs[i] + xs[j]) * cross
        cy += (ys[i] + ys[j]) * cross

    if abs(area) > 1e-15:
        cx /= 6.0 * area
        cy /= 6.0 * area

    return (area, cx, cy)
