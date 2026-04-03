"""Compute pairwise Euclidean distance matrix for 2D points.

Keywords: grpo, numerical, distance, matrix, geometry, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(800,))
def distance_matrix(n: int) -> tuple:
    """Compute full pairwise distance matrix for n deterministic 2D points.

    Returns (sum of all distances, max distance, min nonzero distance).

    Args:
        n: Number of points.

    Returns:
        Tuple of (sum_distances, max_distance, min_nonzero_distance).
    """
    # Generate deterministic points
    xs = [0.0] * n
    ys = [0.0] * n
    for i in range(n):
        xs[i] = (i * 17 + 5) % 1000 * 0.01
        ys[i] = (i * 31 + 11) % 1000 * 0.01

    sum_dist = 0.0
    max_dist = 0.0
    min_dist = 1e18

    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d = (dx * dx + dy * dy) ** 0.5
            sum_dist += d
            if d > max_dist:
                max_dist = d
            if d < min_dist:
                min_dist = d

    return (round(sum_dist, 4), round(max_dist, 4), round(min_dist, 4))
