"""Compute sum of pairwise distances for 3D points.

Keywords: geometry, point, 3d, distance, dataclass, extension type, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


class Point3D:
    """Simple 3D point."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


@python_benchmark(args=(5000,))
def dataclass_point_distance(n: int) -> float:
    """Create n 3D points, compute sum of consecutive pairwise distances.

    Args:
        n: Number of points to create.

    Returns:
        Sum of distances between consecutive point pairs.
    """
    points = [None] * n
    for i in range(n):
        x = ((i * 2654435761) % 10000) / 100.0
        y = ((i * 1664525 + 1013904223) % 10000) / 100.0
        z = ((i * 1103515245 + 12345) % 10000) / 100.0
        points[i] = Point3D(x, y, z)

    total = 0.0
    for i in range(n - 1):
        dx = points[i + 1].x - points[i].x
        dy = points[i + 1].y - points[i].y
        dz = points[i + 1].z - points[i].z
        total += math.sqrt(dx * dx + dy * dy + dz * dz)
    return total
