"""Count points inside a triangle using barycentric coordinates.

Keywords: geometry, point in triangle, barycentric, classification, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(5000000,))
def point_in_triangle(n: int) -> int:
    """Count how many of n test points lie inside a fixed triangle.

    Triangle vertices: (0,0), (100,0), (50,86).
    Points: x[i] = (i*17+3)%200-50, y[i] = (i*13+7)%200-50.
    Uses barycentric coordinate method.

    Args:
        n: Number of test points.

    Returns:
        Count of points inside the triangle.
    """
    # Triangle vertices
    x1, y1 = 0.0, 0.0
    x2, y2 = 100.0, 0.0
    x3, y3 = 50.0, 86.0

    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)

    count = 0
    for i in range(n):
        px = (i * 17 + 3) % 200 - 50
        py = (i * 13 + 7) % 200 - 50

        a = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / denom
        b = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / denom
        c = 1.0 - a - b

        if a >= 0.0 and b >= 0.0 and c >= 0.0:
            count += 1

    return count
