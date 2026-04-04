"""Douglas-Peucker polyline simplification.

Keywords: douglas peucker, polyline, simplification, geometry, line simplify
"""

import math

from cnake_data.benchmarks import python_benchmark


def _point_to_segment_dist_sq(px, py, ax, ay, bx, by):
    """Squared distance from point (px,py) to segment (ax,ay)-(bx,by)."""
    dx = bx - ax
    dy = by - ay

    if dx == 0.0 and dy == 0.0:
        dx = px - ax
        dy = py - ay
        return dx * dx + dy * dy

    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)

    if t < 0.0:
        dx = px - ax
        dy = py - ay
    elif t > 1.0:
        dx = px - bx
        dy = py - by
    else:
        dx = px - (ax + t * dx)
        dy = py - (ay + t * dy)

    return dx * dx + dy * dy


def _douglas_peucker(xs, ys, n, tolerance_sq):
    """Douglas-Peucker simplification using iterative stack approach."""
    if n <= 2:
        return list(range(n))

    markers = [False] * n
    markers[0] = True
    markers[n - 1] = True

    stack = [(0, n - 1)]

    while stack:
        first, last = stack.pop()
        max_dist_sq = 0.0
        index = first

        for i in range(first + 1, last):
            dist_sq = _point_to_segment_dist_sq(
                xs[i], ys[i], xs[first], ys[first], xs[last], ys[last]
            )
            if dist_sq > max_dist_sq:
                max_dist_sq = dist_sq
                index = i

        if max_dist_sq > tolerance_sq:
            markers[index] = True
            stack.append((first, index))
            stack.append((index, last))

    result = []
    for i in range(n):
        if markers[i]:
            result.append(i)
    return result


@python_benchmark(args=(5000,))
def polyline_simplify(n):
    """Generate a noisy polyline of n points and simplify it.

    Args:
        n: Number of points in the polyline.

    Returns:
        Tuple of (num_simplified, total_x, total_y, first_y).
    """
    tolerance = 0.01
    tolerance_sq = tolerance * tolerance

    # Generate deterministic noisy polyline
    xs = [0.0] * n
    ys = [0.0] * n
    for i in range(n):
        xs[i] = i / (n - 1.0) if n > 1 else 0.0
        # Sinusoidal with deterministic noise
        noise = ((i * 7 + 13) % 97) / 97.0 * 0.1 - 0.05
        ys[i] = math.sin(xs[i] * 6.283185307) * 0.5 + noise

    indices = _douglas_peucker(xs, ys, n, tolerance_sq)

    num_simplified = len(indices)
    total_x = 0.0
    total_y = 0.0
    for idx in indices:
        total_x += xs[idx]
        total_y += ys[idx]

    first_y = ys[indices[0]] if indices else 0.0

    return (num_simplified, total_x, total_y, first_y)
