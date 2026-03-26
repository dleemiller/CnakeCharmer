"""
Compute all pairwise Euclidean distances between n 2D points.

Keywords: numerical, euclidean distance, pairwise, geometry, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000,))
def euclidean_distances(n: int) -> float:
    """Compute the sum of all pairwise Euclidean distances between n 2D points.

    Points are generated as (x=i*0.7, y=i*1.3) for i in range(n).
    Returns the sum of distances for all pairs (i, j) where i < j.

    Args:
        n: Number of 2D points.

    Returns:
        Sum of all pairwise Euclidean distances as a float.
    """
    xs = [i * 0.7 for i in range(n)]
    ys = [i * 1.3 for i in range(n)]

    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            total += math.sqrt(dx * dx + dy * dy)

    return total
