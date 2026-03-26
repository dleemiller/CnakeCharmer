"""Layer normalization.

Keywords: layer norm, normalization, neural network, transformer
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def layer_norm(n: int) -> float:
    """Layer normalize groups of 64 values and return sum.

    v[i] = (i * 17 + 5) % 1000 / 10.0

    Args:
        n: Total number of values (must be divisible by 64).

    Returns:
        Sum of all normalized values.
    """
    group_size = 64
    num_groups = n // group_size
    epsilon = 1e-5
    total = 0.0
    for g in range(num_groups):
        offset = g * group_size
        # Compute mean
        mean = 0.0
        for i in range(group_size):
            mean += (((offset + i) * 17 + 5) % 1000) / 10.0
        mean /= group_size
        # Compute variance
        var = 0.0
        for i in range(group_size):
            v = (((offset + i) * 17 + 5) % 1000) / 10.0
            diff = v - mean
            var += diff * diff
        var /= group_size
        inv_std = 1.0 / math.sqrt(var + epsilon)
        # Normalize and accumulate
        for i in range(group_size):
            v = (((offset + i) * 17 + 5) % 1000) / 10.0
            total += (v - mean) * inv_std
    return total
