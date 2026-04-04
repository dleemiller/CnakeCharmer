"""Shoelace polygon area and point-in-polygon count for a regular n-gon.

Keywords: geometry, polygon, winding, shoelace, point-in-polygon, ray casting, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(5000,))
def polygon_winding(n: int) -> tuple:
    """Compute area of a regular n-gon and count how many test points are inside.

    Polygon: regular n-gon with radius 1.0, vertex k at (cos(2πk/n), sin(2πk/n))
    Test points: point i has x=(i*0.618 % 1.0)*2-1, y=(i*0.382 % 1.0)*2-1
    Area via shoelace formula; inside test via ray casting.

    Args:
        n: Number of polygon vertices (and test points).

    Returns:
        Tuple of (area_times_1e6_as_int, inside_count).
    """
    two_pi = 2.0 * math.pi

    # Build vertices
    vx = [math.cos(two_pi * k / n) for k in range(n)]
    vy = [math.sin(two_pi * k / n) for k in range(n)]

    # Shoelace area
    area = 0.0
    for k in range(n):
        kn = (k + 1) % n
        area += vx[k] * vy[kn] - vx[kn] * vy[k]
    area = abs(area) * 0.5

    # Test points and ray casting
    inside_count = 0
    for i in range(n):
        px = (i * 0.618 % 1.0) * 2.0 - 1.0
        py = (i * 0.382 % 1.0) * 2.0 - 1.0
        # Ray casting: count crossings of ray from (px, py) in +x direction
        inside = False
        j = n - 1
        for k in range(n):
            xi = vx[k]
            yi = vy[k]
            xj = vx[j]
            yj = vy[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = k
        if inside:
            inside_count += 1

    return (round(area * 1e6), inside_count)
