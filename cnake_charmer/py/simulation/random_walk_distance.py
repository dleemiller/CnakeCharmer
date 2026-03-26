"""Compute mean squared displacement of deterministic pseudo-random walks.

Keywords: random walk, mean squared displacement, simulation, stochastic, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(2000,))
def random_walk_distance(n: int) -> float:
    """Compute mean squared displacement of n deterministic pseudo-random walks.

    Each walk takes 1000 steps. Step directions are derived from a
    deterministic LCG pseudo-random sequence seeded per walk.

    Args:
        n: Number of walks.

    Returns:
        Mean squared distance across all walks.
    """
    steps = 1000
    total_sq_dist = 0.0

    for w in range(n):
        x = 0
        y = 0
        h = w * 6364136223846793005 + 1
        for _s in range(steps):
            h = (h * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            dx = (h >> 33) % 3 - 1
            h = (h * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            dy = (h >> 33) % 3 - 1
            x += dx
            y += dy
        total_sq_dist += x * x + y * y

    return total_sq_dist / n
