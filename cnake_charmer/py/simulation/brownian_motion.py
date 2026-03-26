"""Simulate Brownian motion for n particles and compute mean squared displacement.

Keywords: brownian motion, simulation, random walk, particle, displacement, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def brownian_motion(n: int) -> float:
    """Simulate n particles doing Brownian motion for 100 steps.

    Each particle starts at (0, 0). At each step, position updates by a
    deterministic pseudo-random displacement based on a hash of (step, particle).
    Returns mean squared displacement from origin.

    Args:
        n: Number of particles.

    Returns:
        Mean squared displacement across all particles.
    """
    steps = 100
    total_sq_disp = 0.0

    for p in range(n):
        x = 0.0
        y = 0.0
        for s in range(steps):
            # Deterministic LCG-style hash
            h = ((s * 6364136223846793005 + p * 1442695040888963407 + 1) >> 16) & 0xFFFFFFFF
            dx = (h % 201 - 100) / 100.0
            h2 = ((h * 6364136223846793005 + 1) >> 16) & 0xFFFFFFFF
            dy = (h2 % 201 - 100) / 100.0
            x += dx
            y += dy
        total_sq_disp += x * x + y * y

    return total_sq_disp / n
