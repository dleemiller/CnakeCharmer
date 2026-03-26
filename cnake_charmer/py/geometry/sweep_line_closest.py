"""Find closest pair distance using sweep line algorithm.

Keywords: closest pair, sweep line, geometry, computational geometry, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def sweep_line_closest(n: int) -> float:
    """Find closest pair distance using sweep line algorithm.

    Points: x[i] = sin(i*0.7)*1000, y[i] = cos(i*1.3)*1000.
    Sorts by x-coordinate, then sweeps with a y-strip.

    Args:
        n: Number of points.

    Returns:
        Minimum distance between any two points as a float.
    """
    # Generate points
    xs = [0.0] * n
    ys = [0.0] * n
    for i in range(n):
        xs[i] = math.sin(i * 0.7) * 1000.0
        ys[i] = math.cos(i * 1.3) * 1000.0

    # Sort by x
    indices = list(range(n))
    indices.sort(key=lambda idx: xs[idx])

    best = 1e18
    # Sweep line: for each point, check subsequent points within x-strip
    for i in range(n):
        pi = indices[i]
        xi = xs[pi]
        yi = ys[pi]
        for j in range(i + 1, n):
            pj = indices[j]
            dx = xs[pj] - xi
            if dx * dx >= best:
                break
            dy = ys[pj] - yi
            dist_sq = dx * dx + dy * dy
            if dist_sq < best:
                best = dist_sq

    return math.sqrt(best)
