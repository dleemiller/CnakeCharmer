"""Numerically stable softmax on groups.

Keywords: softmax, stable, neural network, activation, exp
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def softmax_stable(n: int) -> float:
    """Apply numerically stable softmax on groups of 100 and return sum.

    v[i] = (i * 17 + 5) % 1000 / 100.0
    Sum of all softmax outputs should equal n / 100.

    Args:
        n: Total number of values (must be divisible by 100).

    Returns:
        Sum of all softmax outputs.
    """
    group_size = 100
    num_groups = n // group_size
    total = 0.0
    for g in range(num_groups):
        offset = g * group_size
        # Find max for numerical stability
        max_val = ((offset * 17 + 5) % 1000) / 100.0
        for i in range(1, group_size):
            v = (((offset + i) * 17 + 5) % 1000) / 100.0
            if v > max_val:
                max_val = v
        # Compute exp and sum
        exp_sum = 0.0
        for i in range(group_size):
            v = (((offset + i) * 17 + 5) % 1000) / 100.0
            exp_sum += math.exp(v - max_val)
        # Sum softmax outputs (each group sums to 1.0)
        for i in range(group_size):
            v = (((offset + i) * 17 + 5) % 1000) / 100.0
            total += math.exp(v - max_val) / exp_sum
    return total
