"""Compute sum of nearest-neighbor distances for a set of 2D points.

Keywords: geometry, voronoi, nearest neighbor, distance, brute force, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(3000,))
def voronoi_nearest(n: int) -> float:
    """Compute sum of nearest-neighbor distances for n 2D points.

    Generates n points deterministically:
      x[i] = sin(i * 0.7) * 100
      y[i] = cos(i * 1.3) * 100

    For each point, finds the distance to its nearest neighbor using
    brute-force O(n^2) search, then sums all nearest-neighbor distances.

    Args:
        n: Number of points.

    Returns:
        Sum of nearest-neighbor distances.
    """
    xs = [math.sin(i * 0.7) * 100.0 for i in range(n)]
    ys = [math.cos(i * 1.3) * 100.0 for i in range(n)]

    total = 0.0
    for i in range(n):
        xi = xs[i]
        yi = ys[i]
        min_dist = float("inf")
        for j in range(n):
            if i == j:
                continue
            dx = xi - xs[j]
            dy = yi - ys[j]
            d = math.sqrt(dx * dx + dy * dy)
            if d < min_dist:
                min_dist = d
        total += min_dist

    return total
