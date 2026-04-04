"""Compute total pairwise distances for immutable 2D points.

Keywords: geometry, point, distance, immutable, readonly, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


class ImmutablePoint:
    """2D point with read-only x and y coordinates."""

    __slots__ = ("_x", "_y")

    def __init__(self, x, y):
        object.__setattr__(self, "_x", x)
        object.__setattr__(self, "_y", y)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def __setattr__(self, name, value):
        raise AttributeError("ImmutablePoint attributes are read-only")


@python_benchmark(args=(3000,))
def immutable_point_distance(n: int) -> float:
    """Create n immutable points, compute sum of all pairwise distances.

    Args:
        n: Number of points.

    Returns:
        Sum of Euclidean distances between all pairs.
    """
    points = []
    for i in range(n):
        x = ((i * 2654435761 + 17) % 10000) / 100.0
        y = ((i * 1103515245 + 12345) % 10000) / 100.0
        points.append(ImmutablePoint(x, y))

    total = 0.0
    for i in range(n):
        px = points[i].x
        py_ = points[i].y
        for j in range(i + 1, n):
            dx = px - points[j].x
            dy = py_ - points[j].y
            total += math.sqrt(dx * dx + dy * dy)

    return total
