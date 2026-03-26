"""
Brute-force closest pair of points in 2D.

Keywords: geometry, closest pair, distance, brute force, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(4000,))
def closest_pair_distance(n: int) -> float:
    """Find the minimum Euclidean distance between any two points.

    Generates n 2D points deterministically:
      x[i] = sin(i * 0.7) * 1000
      y[i] = cos(i * 1.3) * 1000

    Uses brute-force O(n^2) comparison of all pairs.

    Args:
        n: Number of points.

    Returns:
        The minimum distance between any two points.
    """
    xs = [math.sin(i * 0.7) * 1000.0 for i in range(n)]
    ys = [math.cos(i * 1.3) * 1000.0 for i in range(n)]

    min_dist = float("inf")
    for i in range(n):
        xi = xs[i]
        yi = ys[i]
        for j in range(i + 1, n):
            dx = xi - xs[j]
            dy = yi - ys[j]
            d = math.sqrt(dx * dx + dy * dy)
            if d < min_dist:
                min_dist = d

    return min_dist
