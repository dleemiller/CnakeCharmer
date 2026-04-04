"""Find the minimum enclosing circle of n deterministic points.

Keywords: geometry, minimum enclosing circle, welzl, bounding circle, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def minimum_enclosing_circle(n: int) -> tuple:
    """Find minimum enclosing circle using incremental algorithm.

    Points: x[i] = ((i*73+11) % 997) - 498.5, y[i] = ((i*37+23) % 991) - 495.5.
    Uses the incremental (non-recursive Welzl-style) algorithm.

    Args:
        n: Number of points.

    Returns:
        Tuple of (center_x, center_y, radius).
    """
    # Generate deterministic points
    px = [0.0] * n
    py = [0.0] * n
    for i in range(n):
        px[i] = ((i * 73 + 11) % 997) - 498.5
        py[i] = ((i * 37 + 23) % 991) - 495.5

    # Shuffle deterministically using Fisher-Yates with fixed seed
    seed = 42
    for i in range(n - 1, 0, -1):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        j = seed % (i + 1)
        px[i], px[j] = px[j], px[i]
        py[i], py[j] = py[j], py[i]

    def _dist(ax, ay, bx, by):
        dx = ax - bx
        dy = ay - by
        return math.sqrt(dx * dx + dy * dy)

    def _circle_from_two(ax, ay, bx, by):
        cx = (ax + bx) * 0.5
        cy = (ay + by) * 0.5
        r = _dist(ax, ay, bx, by) * 0.5
        return cx, cy, r

    def _circle_from_three(ax, ay, bx, by, cx, cy):
        d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-14:
            # Degenerate: pick largest pair
            d1 = _dist(ax, ay, bx, by)
            d2 = _dist(bx, by, cx, cy)
            d3 = _dist(ax, ay, cx, cy)
            if d1 >= d2 and d1 >= d3:
                return _circle_from_two(ax, ay, bx, by)
            elif d2 >= d3:
                return _circle_from_two(bx, by, cx, cy)
            else:
                return _circle_from_two(ax, ay, cx, cy)
        ux = (
            (ax * ax + ay * ay) * (by - cy)
            + (bx * bx + by * by) * (cy - ay)
            + (cx * cx + cy * cy) * (ay - by)
        ) / d
        uy = (
            (ax * ax + ay * ay) * (cx - bx)
            + (bx * bx + by * by) * (ax - cx)
            + (cx * cx + cy * cy) * (bx - ax)
        ) / d
        r = _dist(ax, ay, ux, uy)
        return ux, uy, r

    # Incremental algorithm
    cx, cy, r = px[0], py[0], 0.0

    for i in range(1, n):
        dx = px[i] - cx
        dy = py[i] - cy
        if dx * dx + dy * dy > (r + 1e-10) * (r + 1e-10):
            # Point i is outside current circle
            cx, cy, r = px[i], py[i], 0.0
            for j in range(i):
                dx2 = px[j] - cx
                dy2 = py[j] - cy
                if dx2 * dx2 + dy2 * dy2 > (r + 1e-10) * (r + 1e-10):
                    cx, cy, r = _circle_from_two(px[i], py[i], px[j], py[j])
                    for k in range(j):
                        dx3 = px[k] - cx
                        dy3 = py[k] - cy
                        if dx3 * dx3 + dy3 * dy3 > (r + 1e-10) * (r + 1e-10):
                            cx, cy, r = _circle_from_three(px[i], py[i], px[j], py[j], px[k], py[k])

    return (cx, cy, r)
