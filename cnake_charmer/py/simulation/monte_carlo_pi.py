"""Estimate pi using deterministic quasi-random points.

Keywords: simulation, monte carlo, pi, quasi-random, halton, estimation, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def monte_carlo_pi(n: int) -> tuple:
    """Estimate pi using n deterministic quasi-random points via Halton sequences.

    Uses Halton sequences with bases 2 and 3 for (x, y) in [0,1]^2.
    Points inside the unit circle quadrant contribute to pi estimate.

    Args:
        n: Number of points to sample.

    Returns:
        Tuple of (estimate, points_inside, last_point_distance).
    """
    points_inside = 0
    last_dist = 0.0

    for i in range(1, n + 1):
        # Halton base 2
        x = 0.0
        f = 0.5
        idx = i
        while idx > 0:
            x += f * (idx % 2)
            idx //= 2
            f *= 0.5

        # Halton base 3
        y = 0.0
        f = 1.0 / 3.0
        idx = i
        while idx > 0:
            y += f * (idx % 3)
            idx //= 3
            f /= 3.0

        dist = x * x + y * y
        last_dist = dist
        if dist <= 1.0:
            points_inside += 1

    estimate = 4.0 * points_inside / n

    return (estimate, points_inside, last_dist)
