"""Gaussian kernel density estimation.

Keywords: kernel density, KDE, Gaussian, statistics, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def kernel_density(n: int) -> float:
    """Gaussian kernel density estimation at 100 evaluation points.

    Data: d[i] = (i*17 + 5) % 1000 / 100.0. Bandwidth h=0.5.
    Evaluates density at 100 evenly spaced points from 0 to 10.

    Args:
        n: Number of data points.

    Returns:
        Sum of density estimates at evaluation points as a float.
    """
    h = 0.5
    inv_h = 1.0 / h
    norm_factor = 1.0 / (n * h * math.sqrt(2.0 * math.pi))
    n_eval = 100

    # Generate data
    data = [(i * 17 + 5) % 1000 / 100.0 for i in range(n)]

    total = 0.0
    for ei in range(n_eval):
        x = ei * 10.0 / (n_eval - 1)
        density = 0.0
        for i in range(n):
            u = (x - data[i]) * inv_h
            density += math.exp(-0.5 * u * u)
        total += density * norm_factor

    return total
