"""
Sum of areas of n deterministic 8-sided polygons using the shoelace formula.

Keywords: geometry, polygon, area, shoelace, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def polygon_area(n: int) -> float:
    """Compute the total area of n polygons using the shoelace formula.

    Each polygon i has 8 vertices at angles 2*pi*j/8 with radius
    (j*i + 3) % 50 + 10 for j in range(8).

    Args:
        n: Number of polygons.

    Returns:
        Sum of all polygon areas.
    """
    total_area = 0.0
    two_pi_over_8 = 2.0 * math.pi / 8.0

    for i in range(n):
        # Generate 8 vertices
        vx = [0.0] * 8
        vy = [0.0] * 8
        for j in range(8):
            r = (j * i + 3) % 50 + 10
            angle = j * two_pi_over_8
            vx[j] = r * math.cos(angle)
            vy[j] = r * math.sin(angle)

        # Shoelace formula
        area = 0.0
        for j in range(8):
            j_next = (j + 1) % 8
            area += vx[j] * vy[j_next] - vx[j_next] * vy[j]
        total_area += abs(area) * 0.5

    return total_area
