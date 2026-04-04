"""Compute Lennard-Jones potential energy for particles in 1D.

Keywords: physics, lennard-jones, potential, molecular, pairwise, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(3000,))
def lennard_jones(n: int) -> float:
    """Compute Lennard-Jones potential energy for n particles in 1D.

    x[i] = i*1.0 + sin(i*0.1)*0.1. epsilon=1, sigma=1.
    Sum all pairwise V(r) = 4*eps*((sig/r)^12 - (sig/r)^6).

    Args:
        n: Number of particles.

    Returns:
        Total potential energy as a float.
    """
    epsilon = 1.0
    sigma = 1.0

    x = [i * 1.0 + math.sin(i * 0.1) * 0.1 for i in range(n)]

    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = abs(x[i] - x[j])
            if r > 0:
                sr6 = (sigma / r) ** 6
                sr12 = sr6 * sr6
                total += 4.0 * epsilon * (sr12 - sr6)

    return total
