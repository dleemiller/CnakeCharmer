"""Batch normalization.

Keywords: batch norm, normalization, neural network, statistics
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(2000000,))
def batch_norm(n: int) -> float:
    """Batch normalize n values and return sum of normalized values.

    v[i] = (i * 17 + 5) % 1000 / 10.0
    gamma = 1.0, beta = 0.0, epsilon = 1e-5

    Args:
        n: Number of values.

    Returns:
        Sum of normalized values.
    """
    # Pass 1: compute mean
    mean = 0.0
    for i in range(n):
        mean += ((i * 17 + 5) % 1000) / 10.0
    mean /= n

    # Pass 2: compute variance
    var = 0.0
    for i in range(n):
        v = ((i * 17 + 5) % 1000) / 10.0
        diff = v - mean
        var += diff * diff
    var /= n

    # Normalize and sum
    inv_std = 1.0 / math.sqrt(var + 1e-5)
    total = 0.0
    for i in range(n):
        v = ((i * 17 + 5) % 1000) / 10.0
        total += (v - mean) * inv_std
    return total
