"""Gift wrapping (Jarvis march) convex hull algorithm.

Keywords: geometry, convex hull, gift wrapping, jarvis march, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def gift_wrapping(n: int) -> int:
    """Compute convex hull via gift wrapping and return number of hull vertices.

    Points: x[i] = sin(i*0.7)*100, y[i] = cos(i*1.3)*100.

    Args:
        n: Number of points.

    Returns:
        Number of vertices on the convex hull.
    """
    # Generate points
    xs = [0.0] * n
    ys = [0.0] * n
    for i in range(n):
        xs[i] = math.sin(i * 0.7) * 100.0
        ys[i] = math.cos(i * 1.3) * 100.0

    # Find leftmost point
    start = 0
    for i in range(1, n):
        if xs[i] < xs[start] or (xs[i] == xs[start] and ys[i] < ys[start]):
            start = i

    hull_count = 0
    current = start
    while True:
        hull_count += 1
        candidate = 0
        for i in range(1, n):
            if i == current:
                continue
            if candidate == current:
                candidate = i
                continue
            # Cross product to determine turn direction
            cross = (xs[candidate] - xs[current]) * (ys[i] - ys[current]) - (
                ys[candidate] - ys[current]
            ) * (xs[i] - xs[current])
            if cross < 0:
                candidate = i
            elif cross == 0:
                # Collinear: pick the farther point
                d_cand = (xs[candidate] - xs[current]) ** 2 + (ys[candidate] - ys[current]) ** 2
                d_i = (xs[i] - xs[current]) ** 2 + (ys[i] - ys[current]) ** 2
                if d_i > d_cand:
                    candidate = i

        current = candidate
        if current == start:
            break
        if hull_count > n:
            break

    return hull_count
