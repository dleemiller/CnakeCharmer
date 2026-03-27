"""Sum coordinates of rapidly created/destroyed Point objects.

Keywords: geometry, point, freelist, extension type, allocation, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


class Point:
    """Simple 2D point."""

    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y


@python_benchmark(args=(100000,))
def freelist_point_sum(n: int) -> float:
    """Create n points with deterministic coords, sum all x and y.

    Args:
        n: Number of points to create.

    Returns:
        Sum of all x and y coordinates.
    """
    total = 0.0
    for i in range(n):
        x = ((i * 2654435761) % 100000) / 100.0
        y = ((i * 1664525 + 1013904223) % 100000) / 100.0
        p = Point(x, y)
        total += p.x + p.y
    return total
