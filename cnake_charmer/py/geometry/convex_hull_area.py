"""Compute the area of the convex hull of n 2D points using Graham scan + shoelace formula.

Keywords: convex hull, graham scan, shoelace, geometry, area, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def convex_hull_area(n: int) -> float:
    """Compute the area of the convex hull of n deterministic 2D points.

    Points are generated as x[i] = sin(i*0.7)*100, y[i] = cos(i*1.3)*100.
    Uses Graham scan to find the convex hull, then the shoelace formula for area.

    Args:
        n: Number of points.

    Returns:
        Area of the convex hull as a float.
    """
    # Generate points
    xs = [math.sin(i * 0.7) * 100.0 for i in range(n)]
    ys = [math.cos(i * 1.3) * 100.0 for i in range(n)]

    # Find the bottom-most point (lowest y, then leftmost x)
    pivot = 0
    for i in range(1, n):
        if ys[i] < ys[pivot] or (ys[i] == ys[pivot] and xs[i] < xs[pivot]):
            pivot = i

    # Swap pivot to index 0
    xs[0], xs[pivot] = xs[pivot], xs[0]
    ys[0], ys[pivot] = ys[pivot], ys[0]

    px = xs[0]
    py = ys[0]

    # Sort remaining points by polar angle relative to pivot
    indices = list(range(1, n))
    angles = [math.atan2(ys[i] - py, xs[i] - px) for i in indices]
    # Sort by angle, then by distance for ties
    paired = sorted(
        zip(angles, indices, strict=False),
        key=lambda t: (t[0], (xs[t[1]] - px) ** 2 + (ys[t[1]] - py) ** 2),
    )

    # Graham scan
    stack = [0]
    for _, idx in paired:
        while len(stack) > 1:
            # Cross product to check left turn
            ax = xs[stack[-2]]
            ay = ys[stack[-2]]
            bx = xs[stack[-1]]
            by = ys[stack[-1]]
            cx = xs[idx]
            cy = ys[idx]
            cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if cross <= 0:
                stack.pop()
            else:
                break
        stack.append(idx)

    # Shoelace formula for area
    m = len(stack)
    area = 0.0
    for i in range(m):
        j = (i + 1) % m
        area += xs[stack[i]] * ys[stack[j]]
        area -= xs[stack[j]] * ys[stack[i]]
    area = abs(area) / 2.0

    return area
