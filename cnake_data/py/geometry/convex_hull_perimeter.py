"""Compute the perimeter of the convex hull of n 2D points using Andrew's monotone chain.

Keywords: convex hull, perimeter, geometry, monotone chain, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def convex_hull_perimeter(n: int) -> tuple:
    """Compute the perimeter and vertex count of the convex hull of n deterministic points.

    Points are generated as x[i] = sin(i*0.7)*100, y[i] = cos(i*1.3)*100.
    Uses Andrew's monotone chain to find the convex hull, then sums edge lengths.

    Args:
        n: Number of points.

    Returns:
        Tuple of (perimeter, hull_vertex_count).
    """
    # Generate and sort points
    points = [(math.sin(i * 0.7) * 100.0, math.cos(i * 1.3) * 100.0) for i in range(n)]
    points.sort()

    # Remove duplicates
    unique = [points[0]]
    for i in range(1, len(points)):
        if points[i] != unique[-1]:
            unique.append(points[i])
    points = unique
    m = len(points)

    if m == 1:
        return (0.0, 1)
    if m == 2:
        dx = points[1][0] - points[0][0]
        dy = points[1][1] - points[0][1]
        return (2.0 * math.sqrt(dx * dx + dy * dy), 2)

    # Andrew's monotone chain
    # Lower hull
    lower = []
    for p in points:
        while len(lower) >= 2:
            ox, oy = lower[-2]
            ax, ay = lower[-1]
            if (ax - ox) * (p[1] - oy) - (ay - oy) * (p[0] - ox) <= 0:
                lower.pop()
            else:
                break
        lower.append(p)

    # Upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2:
            ox, oy = upper[-2]
            ax, ay = upper[-1]
            if (ax - ox) * (p[1] - oy) - (ay - oy) * (p[0] - ox) <= 0:
                upper.pop()
            else:
                break
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    h = len(hull)

    # Compute perimeter
    perimeter = 0.0
    for i in range(h):
        j = (i + 1) % h
        dx = hull[j][0] - hull[i][0]
        dy = hull[j][1] - hull[i][1]
        perimeter += math.sqrt(dx * dx + dy * dy)

    return (perimeter, h)
