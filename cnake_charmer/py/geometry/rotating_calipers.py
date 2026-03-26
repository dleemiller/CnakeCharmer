"""Find the diameter of the convex hull of n 2D points using rotating calipers.

Keywords: geometry, convex hull, rotating calipers, diameter, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def rotating_calipers(n: int) -> float:
    """Find the diameter (max pairwise distance) of the convex hull of n points.

    Generates n points deterministically:
      x[i] = sin(i * 0.7) * 100
      y[i] = cos(i * 1.3) * 100

    Builds the convex hull using Andrew's monotone chain algorithm,
    then applies rotating calipers to find the diameter in O(n) on the hull.

    Args:
        n: Number of points.

    Returns:
        The diameter of the convex hull.
    """
    # Generate points
    points = [(math.sin(i * 0.7) * 100.0, math.cos(i * 1.3) * 100.0) for i in range(n)]

    # Sort by x, then y
    points.sort()

    # Remove duplicates
    unique = [points[0]]
    for i in range(1, len(points)):
        if points[i] != unique[-1]:
            unique.append(points[i])
    points = unique

    m = len(points)
    if m == 1:
        return 0.0
    if m == 2:
        dx = points[1][0] - points[0][0]
        dy = points[1][1] - points[0][1]
        return math.sqrt(dx * dx + dy * dy)

    # Andrew's monotone chain convex hull
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

    # Concatenate, removing last point of each half (it's repeated)
    hull = lower[:-1] + upper[:-1]
    h = len(hull)

    if h <= 1:
        return 0.0
    if h == 2:
        dx = hull[1][0] - hull[0][0]
        dy = hull[1][1] - hull[0][1]
        return math.sqrt(dx * dx + dy * dy)

    # Rotating calipers
    max_dist_sq = 0.0
    j = 1
    for i in range(h):
        ni = (i + 1) % h
        while True:
            nj = (j + 1) % h
            # Cross product to check if we should advance j
            eix = hull[ni][0] - hull[i][0]
            eiy = hull[ni][1] - hull[i][1]
            ejx = hull[nj][0] - hull[j][0]
            ejy = hull[nj][1] - hull[j][1]
            cross = eix * ejy - eiy * ejx
            if cross > 0:
                j = nj
            else:
                break

        dx = hull[i][0] - hull[j][0]
        dy = hull[i][1] - hull[j][1]
        dist_sq = dx * dx + dy * dy
        if dist_sq > max_dist_sq:
            max_dist_sq = dist_sq

    return math.sqrt(max_dist_sq)
