"""Count how many test points lie inside a regular 12-gon using ray casting.

Keywords: point in polygon, ray casting, geometry, containment, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def point_in_polygon(n: int) -> int:
    """Count how many of n test points lie inside a regular 12-gon.

    The polygon is a regular 12-gon of radius 50 centered at the origin.
    Test points: x[i] = (i*17+3)%200 - 100, y[i] = (i*13+7)%200 - 100.
    Uses the ray casting algorithm.

    Args:
        n: Number of test points.

    Returns:
        Count of points inside the polygon.
    """
    # Build the 12-gon vertices
    num_verts = 12
    poly_x = [50.0 * math.cos(2.0 * math.pi * k / num_verts) for k in range(num_verts)]
    poly_y = [50.0 * math.sin(2.0 * math.pi * k / num_verts) for k in range(num_verts)]

    count = 0
    for i in range(n):
        tx = (i * 17 + 3) % 200 - 100
        ty = (i * 13 + 7) % 200 - 100

        # Ray casting: count intersections with polygon edges
        inside = False
        j = num_verts - 1
        for k in range(num_verts):
            yi = poly_y[k]
            yj = poly_y[j]
            xi = poly_x[k]
            xj = poly_x[j]

            if ((yi > ty) != (yj > ty)) and (tx < (xj - xi) * (ty - yi) / (yj - yi) + xi):
                inside = not inside
            j = k

        if inside:
            count += 1

    return count
